import os
import torch
from transformers import AutoModel, AutoTokenizer

from .soft_prompt import SoftPromptModule
from .student_wrapper import StudentWithSoftPrompt


MASK_TOKEN_ID = 126336


@torch.no_grad()
def onestep_generate(model_name: str, ckpt_path: str, prompt_text: str, gen_length: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    # 恢复 soft prompt
    if hasattr(base, "get_input_embeddings"):
        hidden = base.get_input_embeddings().weight.shape[-1]
    elif hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
        hidden = base.model.embed_tokens.weight.shape[-1]
    else:
        raise ValueError("Cannot infer base model hidden size")
    soft_prompt = SoftPromptModule(hidden_size=hidden)
    state = torch.load(ckpt_path, map_location=device)
    # 兼容 Lightning 保存
    if "state_dict" in state:
        sd = {k.replace("student.soft_prompt.", ""): v for k, v in state["state_dict"].items() if k.startswith("student.soft_prompt.")}
    else:
        sd = state
    soft_prompt.load_state_dict(sd, strict=False)
    soft_prompt = soft_prompt.to(device).eval()

    student = StudentWithSoftPrompt(base, tokenizer, soft_prompt).to(device).eval()

    # 构造初始全掩码序列 x1，r=0, t=1
    tpl = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(tpl)["input_ids"]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    x = torch.full((1, input_ids.shape[1] + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :input_ids.shape[1]] = input_ids

    r = torch.zeros(x.size(0), device=device)
    t = torch.ones(x.size(0), device=device)

    u_logits = student(x, r, t)  # (B, L, V)
    out = torch.argmax(u_logits, dim=-1)

    text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return text


def main():
    model = os.environ.get("LLADA_MODEL", "GSAI-ML/LLaDA-8B-Instruct")
    ckpt = os.environ.get("SOFT_PROMPT_CKPT", "./checkpoints_meanflow_llm/last.ckpt")
    q = os.environ.get("QUESTION", "What is the capital of France?")
    print(onestep_generate(model, ckpt, q))


if __name__ == "__main__":
    main()

