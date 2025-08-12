import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from .student_wrapper import StudentWithSoftPrompt
from .jvp_utils import jvp_with_embedding


MASK_TOKEN_ID = 126336


class LLaDAStudentLightning(pl.LightningModule):
    """
    Lightning 模块：实现基于 MeanFlow 恒等式的离散蒸馏训练。
    仅训练 SoftPrompt 模块参数，教师与学生基座权重冻结。
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        soft_prompt_module: nn.Module,
        lr: float = 1e-4,
        kl_weight: float = 1.0,
        use_autograd_jvp: bool = True,
        teacher_dtype: torch.dtype = torch.float16,
        jvp_topk: int = 32,
        jvp_on_mask_only: bool = True,
        du_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["base_model", "tokenizer", "soft_prompt_module"])  # 便于 ckpt

        # 冻结教师与学生基座
        self.teacher = base_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.base = base_model.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.student = StudentWithSoftPrompt(self.base, tokenizer, soft_prompt_module)
        self.lr = lr
        self.kl_weight = kl_weight
        self.use_autograd_jvp = use_autograd_jvp
        self.teacher_dtype = teacher_dtype
        self.jvp_topk = int(jvp_topk)
        self.jvp_on_mask_only = bool(jvp_on_mask_only)
        # 有限差分步长（论文中的 sg 目标）
        self.du_epsilon = float(du_epsilon)

        # 供 embedding 空间计算使用
        if hasattr(self.base, "get_input_embeddings"):
            self.embedding_layer = self.base.get_input_embeddings()
        elif hasattr(self.base, "model") and hasattr(self.base.model, "embed_tokens"):
            self.embedding_layer = self.base.model.embed_tokens
        else:
            raise ValueError("Cannot locate input embedding layer in base model")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.student.soft_prompt.parameters(), lr=self.lr)
        return optim

    @torch.no_grad()
    def _mask_data(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 与 GUIDELINES 中一致：随机按 p_mask 比例置为 MASK_TOKEN_ID
        b, l = input_ids.shape
        t = t.view(-1, 1).repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids.device) < t
        noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)
        return noisy_batch

    def _sample_t_r(self, batch_size: int, device: torch.device):
        t = torch.rand(batch_size, device=device)
        r = torch.rand(batch_size, device=device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        # flow_ratio: 一定比例取 r=t
        flow_ratio = 0.5
        num_eq = int(flow_ratio * batch_size)
        if num_eq > 0:
            idx = torch.randperm(batch_size, device=device)[:num_eq]
            r[idx] = t[idx]
        return t, r

    def _student_fn_embed(self, xt_embed: torch.Tensor, r: torch.Tensor, t: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 用 student 的 soft prompt 重构 inputs_embeds，然后走 base 输入
        bsz = xt_embed.size(0)
        prompts = self.student.soft_prompt(r, t)  # (B, P, H)
        inputs_embeds = torch.cat([prompts, xt_embed], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.ones(bsz, prompts.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # 强制使用 math SDP 内核，避免 Flash 内核在 forward-AD/二阶导时不支持
        try:
            from torch.backends.cuda import sdp_kernel
            if torch.cuda.is_available():
                with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    outputs = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                outputs = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        except Exception:
            outputs = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, prompts.size(1):]
        return logits

    def training_step(self, batch, batch_idx):
        input_ids: torch.Tensor = batch["input_ids"].to(self.device)
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        bsz = input_ids.size(0)
        t, r = self._sample_t_r(bsz, input_ids.device)
        xt = self._mask_data(input_ids, t)

        # 教师瞬时流 (logits)
        with torch.no_grad():
            # 教师半精度推理，节省显存
            v_logits = self.teacher(xt).logits.to(self.teacher_dtype).to(torch.float32)

        # Step 1) 计算预测 u_theta（需保留梯度）
        xt_embed = self.embedding_layer(xt)
        vocab_embeds = self.embedding_layer.weight  # (V, H)
        # Top-K 稀疏方向，避免分配 (B,L,V) 概率张量
        k = max(1, min(self.jvp_topk, v_logits.shape[-1]))
        topk_vals, topk_idx = torch.topk(v_logits, k=k, dim=-1)  # (B,L,K)
        sparse_weights = F.softmax(topk_vals.to(torch.float32), dim=-1).to(topk_vals.dtype)  # (B,L,K)
        # 取出对应词向量并加权求和
        # embeddings: (B,L,K,H)
        embeddings = F.embedding(topk_idx, vocab_embeds)  # gather by indices
        target_embed = (sparse_weights.unsqueeze(-1) * embeddings).sum(dim=-2)  # (B,L,H)
        tangent_xt = target_embed - xt_embed
        tangent_t = torch.ones_like(t)

        # 仅对 [MASK] 位置做 JVP 的方向（其余位置置零）
        if self.jvp_on_mask_only:
            mask_positions = (xt == MASK_TOKEN_ID).unsqueeze(-1)  # (B,L,1)
            tangent_xt = tangent_xt * mask_positions.to(tangent_xt.dtype)

        # 预测 u_theta（带梯度）
        u_logits = self._student_fn_embed(xt_embed, r, t, attention_mask)

        # Step 2) 用有限差分计算 du/dt（目标项，停止梯度）
        eps = self.du_epsilon
        t_eps = torch.clamp(t + eps, max=1.0)
        with torch.no_grad():
            u_logits_eps = self._student_fn_embed(xt_embed + eps * tangent_xt, r, t_eps, attention_mask)
        du_dt_logits = (u_logits_eps - u_logits.detach()) / eps

        # 构建目标 u_tgt_logits = v_logits - (t-r) * du/dt
        scale = (t - r).view(-1, 1, 1)
        with torch.no_grad():
            u_tgt_logits = v_logits - scale * du_dt_logits

        # KL 散度，仅在 mask 位置
        mask_positions = (xt == MASK_TOKEN_ID)
        log_probs_student = F.log_softmax(u_logits, dim=-1)
        probs_target = F.softmax(u_tgt_logits, dim=-1)
        loss = F.kl_div(
            log_probs_student[mask_positions],
            probs_target[mask_positions],
            reduction="batchmean",
            log_target=False,
        )
        self.log("train/kl_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

