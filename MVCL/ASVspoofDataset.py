import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

class ASVspoofDataset(Dataset):
    def __init__(self, list_path, max_len=64000):
        """
        list_path: 之前生成的 txt 列表文件的路径 (例如 'data_lists/eval_list.txt')
        max_len: 音频截断/填充长度 (64000 采样点约等于 4秒)
        """
        self.data_list = []
        self.max_len = max_len
        
        # 读取 txt 列表
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                # 格式: /path/to/file.flac label
                path = parts[0]
                label = int(parts[1])
                self.data_list.append((path, label))
                
        print(f"✅ 已加载数据集列表: {list_path}")
        print(f"   样本数量: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, label = self.data_list[idx]
        
        # 1. 加载音频
        # torchaudio.load 返回 (channels, time)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 2. 长度处理 (固定长度)
        # 如果音频太短，补零；如果太长，截断
        curr_len = waveform.shape[1]
        if curr_len < self.max_len:
            # 补零 (Padding)
            pad_width = self.max_len - curr_len
            # pad 最后一个维度 (time) 的右侧
            waveform = torch.nn.functional.pad(waveform, (0, pad_width))
        else:
            # 截断 (Truncating)
            waveform = waveform[:, :self.max_len]
            
        # 3. 返回 demo 模型需要的字典格式
        return {
            "audio": waveform,      # Shape: (1, 64000)
            "label": label,         # int: 0 or 1
            "sample_rate": sample_rate
        }
    




