import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class PackedIdsDataset(Dataset):
    """
    简单数据集：输入为预打包的 token id 序列列表。
    每个样本字典包含: input_ids, attention_mask
    """

    def __init__(self, sequences: List[torch.Tensor], pad_id: int = 0, max_len: int = 4096) -> None:
        self.pad_id = pad_id
        self.max_len = max_len
        self.data = []
        for s in sequences:
            s = s[:max_len]
            self.data.append(s.clone())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ids = self.data[idx]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


def create_dataloader(sequences: List[torch.Tensor], batch_size: int, pad_id: int = 0, max_len: int = 4096, shuffle: bool = True):
    ds = PackedIdsDataset(sequences, pad_id=pad_id, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

