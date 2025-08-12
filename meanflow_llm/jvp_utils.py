import torch
from typing import Callable, Tuple


def jvp_with_embedding(
    fn_embed: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xt_embed: torch.Tensor,
    t: torch.Tensor,
    tangent_xt_embed: torch.Tensor,
    tangent_t: torch.Tensor,
    use_autograd: bool = True,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 embedding 空间执行 JVP：返回 u 与 du/dt。
    fn_embed 接受 (xt_embed, t) -> logits。
    针对 Flash-Attn 前/后向 AD 不支持的问题，统一强制使用 math SDP kernel。
    """
    # 选择统一的 sdpa kernel 上下文
    def sdpa_ctx():
        try:
            from torch.nn.attention import sdpa_kernel
            return sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    if use_autograd:
        # 不构建 du 的反向图，显著降低显存；u 仍然带梯度用于训练
        with sdpa_ctx():
            u, du_dt = torch.autograd.functional.jvp(
                lambda _xt_e, _t: fn_embed(_xt_e, _t),
                (xt_embed, t),
                (tangent_xt_embed, tangent_t),
                create_graph=create_graph,
            )
        return u, du_dt
    else:
        from torch.func import jvp
        # forward-mode jvp 一般更省显存
        with sdpa_ctx():
            u, du_dt = jvp(
                lambda _xt_e, _t: fn_embed(_xt_e, _t),
                (xt_embed, t),
                (tangent_xt_embed, tangent_t),
            )
        return u, du_dt

