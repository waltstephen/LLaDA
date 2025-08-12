import torch
import torch.nn as nn
from typing import Optional


class StudentWithSoftPrompt(nn.Module):
    """
    将冻结的 LLaDA 基座模型与 SoftPrompt 模块包装为一个新前向：
    forward(input_ids, r, t) -> logits

    通过将 soft prompts 拼接在输入 embedding 的前部，实现 prefix-tuning 效果，
    再走模型原始的 forward 计算 logits。
    """

    def __init__(self, base_model: nn.Module, tokenizer, soft_prompt_module: nn.Module):
        super().__init__()
        self.base = base_model
        self.tokenizer = tokenizer
        self.soft_prompt = soft_prompt_module

        for p in self.base.parameters():
            p.requires_grad = False

        # 推断隐藏维度
        if hasattr(self.base, "get_input_embeddings"):
            self.hidden_size = self.base.get_input_embeddings().weight.shape[-1]
        elif hasattr(self.base, "model") and hasattr(self.base.model, "embed_tokens"):
            self.hidden_size = self.base.model.embed_tokens.weight.shape[-1]
        else:
            raise ValueError("Cannot infer embedding hidden size from the base model")

    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base, "get_input_embeddings"):
            emb = self.base.get_input_embeddings()(input_ids)
        elif hasattr(self.base, "model") and hasattr(self.base.model, "embed_tokens"):
            emb = self.base.model.embed_tokens(input_ids)
        else:
            raise ValueError("Base model has no known embedding accessor")
        return emb

    def _forward_from_embedding(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 大多数 HuggingFace 模型支持 inputs_embeds 作为输入
        outputs = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    def forward(self, input_ids: torch.Tensor, r: torch.Tensor, t: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (B, L)
            r: (B,)
            t: (B,)
        Returns:
            logits: (B, L + P, V) 但我们仅取后面的 L 部分作为与输入 token 对齐的输出
        """
        device = input_ids.device
        prompts = self.soft_prompt(r, t)  # (B, P, H)

        token_embeds = self._embed_input_ids(input_ids)  # (B, L, H)
        inputs_embeds = torch.cat([prompts, token_embeds], dim=1)  # (B, P+L, H)

        if attention_mask is not None:
            # 在前缀位置补 1
            bsz = attention_mask.size(0)
            prefix_mask = torch.ones(bsz, prompts.size(1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        logits = self._forward_from_embedding(inputs_embeds, attention_mask=attention_mask)
        # 丢弃前缀对应的 logits
        logits = logits[:, prompts.size(1):]
        return logits

