import os
import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

from .soft_prompt import SoftPromptModule
from .lightning_module import LLaDAStudentLightning
from .data import create_dataloader


def main():
    model_name = os.environ.get("LLADA_MODEL", "GSAI-ML/LLaDA-8B-Base")
    lr = float(os.environ.get("LR", 1e-4))
    prompt_length = int(os.environ.get("PROMPT_LENGTH", 16))
    max_len = int(os.environ.get("MAX_LEN", 2048))
    batch_size = int(os.environ.get("BATCH_SIZE", 1))
    steps = int(os.environ.get("STEPS", 1000))
    use_autograd = bool(int(os.environ.get("USE_AUTOGRAD", 1)))
    out_dir = os.environ.get("OUT_DIR", "./checkpoints_meanflow_llm")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    # 推断隐藏维度
    if hasattr(base, "get_input_embeddings"):
        hidden = base.get_input_embeddings().weight.shape[-1]
    elif hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
        hidden = base.model.embed_tokens.weight.shape[-1]
    else:
        raise ValueError("Cannot infer base model hidden size")

    soft_prompt = SoftPromptModule(hidden_size=hidden, prompt_length=prompt_length)

    lightning_module = LLaDAStudentLightning(
        base_model=base,
        tokenizer=tokenizer,
        soft_prompt_module=soft_prompt,
        lr=lr,
        use_autograd_jvp=use_autograd,
    )

    # 这里用一个玩具数据：重复若干条 <BOS> + <EOS>，以便跑通训练逻辑
    # 实际使用时应替换为真实预训练数据集
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 1
    eos = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
    toy = torch.tensor([bos, eos] + [eos] * (max_len - 2), dtype=torch.long)
    data_path = os.environ.get("DATA_PATH", "")
    sequences = []
    if data_path and os.path.isfile(data_path):
        # 从纯文本文件加载，每行一条样本；或从包含 JSONL 的文件中字段 text
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") and line.endswith("}"):
                    try:
                        import json
                        obj = json.loads(line)
                        text = obj.get("text", "")
                    except Exception:
                        text = line
                else:
                    text = line
                if not text:
                    continue
                ids = tokenizer(text)["input_ids"]
                sequences.append(torch.tensor(ids, dtype=torch.long))
    else:
        # toy data fallback
        sequences = [toy.clone() for _ in range(256)]
    train_loader = create_dataloader(sequences, batch_size=batch_size, max_len=max_len, shuffle=True)

    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=out_dir, save_top_k=1, monitor="train/kl_loss", mode="min")
    trainer = pl.Trainer(
        max_steps=steps,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        log_every_n_steps=5,
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()

