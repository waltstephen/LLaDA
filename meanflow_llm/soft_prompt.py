import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    将标量 r, t 编码为高维向量。
    简单实现：Fourier 特征 + MLP。
    """

    def __init__(self, embed_dim: int = 128, fourier_dim: int = 32):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.linspace(-3.0, 3.0, fourier_dim)
            ),
            persistent=False,
        )
        in_dim = fourier_dim * 4  # sin/cos for r and t
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: (B,) in [0,1]
            t: (B,) in [0,1]
        Returns:
            time_emb: (B, D)
        """
        # shape to (B, 1)
        r = r.view(-1, 1)
        t = t.view(-1, 1)

        # (B, F)
        r_w = r * self.freqs.view(1, -1)
        t_w = t * self.freqs.view(1, -1)
        r_feats = torch.cat([torch.sin(r_w), torch.cos(r_w)], dim=-1)
        t_feats = torch.cat([torch.sin(t_w), torch.cos(t_w)], dim=-1)
        stacked = torch.cat([r_feats, t_feats], dim=-1)
        return self.mlp(stacked)


class SoftPromptModule(nn.Module):
    """
    软提示模块（Prefix-Tuning 样式，基于输入嵌入级别）。
    - 根据 (r, t) 生成可训练的虚拟 token 嵌入，前缀拼接到原输入嵌入前。
    - 不需要侵入式修改底层 Transformer 的 KV Cache。
    """

    def __init__(
        self,
        hidden_size: int,
        prompt_length: int = 16,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.time_encoder = TimeEncoder(embed_dim=time_embed_dim)
        # 低参数量：单层线性直接生成 (P*H)，避免 (P*H)^2 巨大参数造成 OOM
        self.prompt_linear = nn.Linear(time_embed_dim, hidden_size * prompt_length)
        # 基础可学习前缀，提供稳态先验
        self.base_prompt = nn.Parameter(torch.zeros(prompt_length, hidden_size))

        # 初始化为小幅度值，稳定训练
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.base_prompt, mean=0.0, std=0.02)

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: (B,)
            t: (B,)
        Returns:
            prompts: (B, P, H)
        """
        time_emb = self.time_encoder(r, t)  # (B, D)
        prompt_flat = self.prompt_linear(time_emb)  # (B, P*H)
        prompts_delta = prompt_flat.view(-1, self.prompt_length, self.hidden_size)
        prompts = prompts_delta + self.base_prompt.unsqueeze(0).to(prompts_delta.dtype)
        return prompts

