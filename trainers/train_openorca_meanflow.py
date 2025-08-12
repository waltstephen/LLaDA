import os
import argparse
import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning.loggers import Logger

from meanflow_llm.soft_prompt import SoftPromptModule
from meanflow_llm.lightning_module import LLaDAStudentLightning
from datasets import create_openorca_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, required=True, help="OpenOrca parquet 文件路径")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--out", type=str, default="./checkpoints_meanflow_llm_openorca")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--prompt_length", type=int, default=16)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000, help="总训练步数；>0 时按步数停止")
    p.add_argument("--max_epochs", type=int, default=-1, help="最大 epoch 数；>0 时按 epoch 停止")
    p.add_argument("--steps_per_epoch", type=int, default=-1, help="每个 epoch 的训练步数；>0 时覆盖 Lightning 的自动长度")
    p.add_argument("--precision", type=str, default="bf16-mixed", help="Lightning 精度：bf16-mixed/fp16-mixed/32-true 等")
    p.add_argument("--accumulate", type=int, default=1, help="梯度累积步数")
    p.add_argument("--autograd_jvp", action="store_true")
    p.add_argument("--jvp_topk", type=int, default=32, help="JVP 方向的 Top-K 稀疏聚合（按 logits topk）")
    p.add_argument("--jvp_on_mask_only", type=int, default=1, help="仅对 [MASK] 位置计算 JVP (1/0)")
    p.add_argument("--enable_ckpt", type=int, default=1, help="是否启用激活检查点 (1/0)")
    p.add_argument("--du_epsilon", type=float, default=1e-3, help="有限差分步长 eps")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def _resolve_model_path(model_arg: str) -> str:
    """支持三种形式：
    - HuggingFace repo id: e.g. 'GSAI-ML/LLaDA-8B-Base'
    - 本地快照目录: e.g. '~/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/<rev>'
    - 本地模型根目录: 含 tokenizer_config.json 等文件
    """
    model_arg = os.path.expanduser(model_arg)
    if os.path.isdir(model_arg):
        snapshots_dir = os.path.join(model_arg, "snapshots")
        if os.path.isdir(snapshots_dir):
            candidates = [
                os.path.join(snapshots_dir, d)
                for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            ]
            if len(candidates) > 0:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]
        return model_arg
    return model_arg


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        pass

    model_path = _resolve_model_path(args.model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 在 CPU 上加载，交由 Lightning 移动到对应 rank 的 GPU；优先 bf16，其次 fp16
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).eval()
    # 可选启用激活检查点以降低显存；需关闭 use_cache
    if args.enable_ckpt:
        try:
            if hasattr(base, 'config') and hasattr(base.config, 'use_cache'):
                base.config.use_cache = False
            if hasattr(base, 'gradient_checkpointing_enable'):
                base.gradient_checkpointing_enable()
        except Exception:
            pass

    if hasattr(base, "get_input_embeddings"):
        hidden = base.get_input_embeddings().weight.shape[-1]
    elif hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
        hidden = base.model.embed_tokens.weight.shape[-1]
    else:
        raise ValueError("Cannot infer base model hidden size")

    soft_prompt = SoftPromptModule(hidden_size=hidden, prompt_length=args.prompt_length)

    lightning_module = LLaDAStudentLightning(
        base_model=base,
        tokenizer=tokenizer,
        soft_prompt_module=soft_prompt,
        lr=args.lr,
        use_autograd_jvp=args.autograd_jvp,
        teacher_dtype=torch.float16,
        jvp_topk=args.jvp_topk,
        jvp_on_mask_only=bool(args.jvp_on_mask_only),
        du_epsilon=args.du_epsilon,
    )

    train_loader = create_openorca_dataloader(
        parquet_path=args.parquet,
        tokenizer_name=model_path,
        batch_size=args.batch_size,
        max_length=args.max_len,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=args.out, save_top_k=1, monitor="train/kl_loss", mode="min")
    # SwanLab logger（国内可用）。参考文档: [SwanLab Docs](https://docs.swanlab.cn/en/)
    class SwanLabLogger(Logger):
        def __init__(self, project: str, name: str | None, save_dir: str):
            super().__init__()
            import swanlab
            self._swanlab = swanlab
            self._run = swanlab.init(project=project, experiment_name=name, work_dir=save_dir)

        @property
        def name(self):
            return "swanlab"

        @property
        def version(self):
            return self._run.id if hasattr(self._run, "id") else None

        def log_metrics(self, metrics, step=None):
            # 只在主进程记录
            if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._swanlab.log({k: v}, step=step)

        def log_hyperparams(self, params):
            pass

        def finalize(self, status):
            try:
                self._swanlab.finish()
            except Exception:
                pass

    def _is_rank_zero() -> bool:
        # 优先用 torch.distributed；否则退化到环境变量；最后单机 True
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        if rank_env is not None:
            try:
                return int(rank_env) == 0
            except Exception:
                return True
        return True

    logger = False
    if _is_rank_zero():
        logger = SwanLabLogger(
            project=os.environ.get("SWANLAB_PROJECT", "llada-meanflow"),
            name=os.environ.get("SWANLAB_RUN_NAME", None),
            save_dir=args.out,
        )
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    trainer_kwargs = dict(
        accelerator="gpu" if num_devices > 0 else "cpu",
        devices=num_devices if num_devices > 0 else 1,
        strategy="ddp" if num_devices > 1 else None,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=5,
    )
    if args.max_steps and args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.max_epochs and args.max_epochs > 0:
        trainer_kwargs["max_epochs"] = args.max_epochs
    if args.steps_per_epoch and args.steps_per_epoch > 0:
        trainer_kwargs["limit_train_batches"] = args.steps_per_epoch

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(lightning_module, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()

