import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
import re

class VocabularyBuilder:
    """Builds and manages vocabulary for the dataset"""
    
    def __init__(self, min_freq: int = 2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()
        self.min_freq = min_freq
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts"""
        # Count word frequencies
        for text in texts:
            words = self._tokenize(text)
            self.word_freq.update(words)
        
        # Add words that meet minimum frequency
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def encode(self, text: str, max_length: int) -> List[int]:
        """Convert text to sequence of indices"""
        words = self._tokenize(text)
        sequence = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate to max_length
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))  # 0 is <PAD>
        else:
            sequence = sequence[:max_length]
            
        return sequence
    
    def decode(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text"""
        words = [self.idx2word.get(idx, "<UNK>") for idx in sequence if idx != 0]
        return " ".join(words)
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

class AnimeDataset(Dataset):
    """Custom dataset for anime chatbot training"""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 vocab_builder: VocabularyBuilder,
                 max_length: int = 100):
        """
        Initialize dataset
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            vocab_builder: VocabularyBuilder instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab_builder = vocab_builder
        self.max_length = max_length
        
        # Build vocabulary if not already built
        if vocab_builder.vocab_size == 2:  # Only <PAD> and <UNK>
            vocab_builder.build_vocab(texts)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence
        sequence = self.vocab_builder.encode(text, self.max_length)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for DataLoader"""
        sequences, labels = zip(*batch)
        return torch.stack(sequences), torch.stack(labels)