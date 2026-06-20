from abc import ABC, abstractmethod
from datasets import load_from_disk
import os

class BaseReasoningDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx) -> dict:
        pass

class GSM8KDataset(BaseReasoningDataset):
    def __init__(self, data_path="data/raw/gsm8k", split="train"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py first.")
        self.dataset = load_from_disk(data_path)[split]
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]
        return {
            'question': item['question'],
            'reasoning': item['answer'].split('####')[0].strip(),
            'answer': item['answer'].split('####')[-1].strip(),
            'is_correct': True,
            'error_type': None,
            'problem_domain': 'gsm8k'
        }
