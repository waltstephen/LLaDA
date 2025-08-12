import os
import json
import bisect
import pyarrow.parquet as pq
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class OpenOrcaParquetDataset(Dataset):
    """
    直接基于 OpenOrca parquet 文件的流式读取数据集。
    - 字段包含：system_prompt, question, response。
    - 组装为 Instruct 风格提示 -> token ids。
    - 支持子样本采样（max_samples）。
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        chat_template: bool = True,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.chat_template = chat_template

        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        self._pf = pq.ParquetFile(parquet_path)
        meta = self._pf.metadata
        self._num_rows_total = meta.num_rows
        # row group starts (prefix sum)
        self._rg_sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        self._rg_starts = []
        acc = 0
        for n in self._rg_sizes:
            self._rg_starts.append(acc)
            acc += n
        self._num_rows = self._num_rows_total if max_samples is None else min(self._num_rows_total, max_samples)

    def __len__(self):
        return self._num_rows

    def _format_prompt(self, system_prompt: str, question: str) -> str:
        if not self.chat_template:
            return f"<system>\n{system_prompt}\n</system>\n<user>\n{question}\n</user>\n<assistant>\n"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return text

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._num_rows:
            raise IndexError

        # 定位到 row group 与组内索引
        # rg_idx 满足 rg_starts[rg_idx] <= idx < rg_starts[rg_idx] + rg_sizes[rg_idx]
        rg_idx = bisect.bisect_right(self._rg_starts, idx) - 1
        if rg_idx < 0:
            rg_idx = 0
        row_in_group = idx - self._rg_starts[rg_idx]

        table = self._pf.read_row_group(rg_idx, columns=None)

        def get_cell(name: str):
            if name in table.column_names:
                return table.column(name)[row_in_group].as_py()
            return ""

        system_prompt = get_cell("system_prompt") or ""
        question = get_cell("question") or ""
        response = get_cell("response") or ""

        prompt_text = self._format_prompt(system_prompt, question)
        # 拼接答案用于 teacher 强化 logit 一致性（不用于 student 目标）
        full_text = prompt_text + response

        ids = self.tokenizer(full_text, truncation=True, max_length=self.max_length)["input_ids"]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_pad(batch, pad_id: int):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attn = []
    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        if ids.size(0) < max_len:
            pad_len = max_len - ids.size(0)
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)], dim=0)
        input_ids.append(ids)
        attn.append(mask)
    return {"input_ids": torch.stack(input_ids, 0), "attention_mask": torch.stack(attn, 0)}


def create_openorca_dataloader(
    parquet_path: str,
    tokenizer_name: str = "GSAI-ML/LLaDA-8B-Base",
    batch_size: int = 1,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    num_workers: int = 2,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    ds = OpenOrcaParquetDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        chat_template=True,
    )
    pad_id = tokenizer.pad_token_id or 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, pad_id),
        drop_last=True,
    )

