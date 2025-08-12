## LLaDA-MeanFlow (1-Step) 训练与推理

本目录提供基于 Prompt Tuning 与 MeanFlow 恒等式的离散蒸馏实现，将多步扩散 LLaDA 蒸馏为 1-Step 生成。

### 依赖安装

```bash
pip install -r requirements.txt
```

确保已安装 `torch>=2.4`，`transformers==4.38.2`，`pytorch-lightning>=2.3`，`einops`。

### 训练

```bash
export LLADA_MODEL=GSAI-ML/LLaDA-8B-Base
export DATA_PATH=/path/to/text_or_jsonl
export OUT_DIR=./checkpoints_meanflow_llm
python -m meanflow_llm.train_meanflow_llm
```

可选环境变量：
- `LR`、`PROMPT_LENGTH`、`MAX_LEN`、`BATCH_SIZE`、`STEPS`、`USE_AUTOGRAD`。

说明：
- 仅训练 `SoftPromptModule` 参数，教师与学生基座参数冻结。
- 损失为 KL(student(u) || target(v - (t-r)du/dt))，JVP 在 embedding 空间计算。

### 1-Step 推理

```bash
export LLADA_MODEL=GSAI-ML/LLaDA-8B-Instruct
export SOFT_PROMPT_CKPT=./checkpoints_meanflow_llm/last.ckpt
export QUESTION="What is the capital of France?"
python -m meanflow_llm.infer_onestep
```

或在代码中调用 `onestep_generate()`。

