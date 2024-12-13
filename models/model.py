import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based architecture
    Adds information about token position in sequence
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:x.size(1)]

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing dependencies between words
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Linear projections and split into heads
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.output_layer(context)
        return output

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward networks
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = SelfAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AnimeChatbotModel(nn.Module):
    """
    Main model architecture combining embedding, transformer encoder, and classification layers
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_encoder_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 512):
        """
        Initialize the model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_classes: Number of output classes
            num_encoder_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Project embeddings to hidden dimension if different
        self.input_projection = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Output layers
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.final_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of token indices (batch_size, seq_len)
            
        Returns:
            Output tensor of class logits (batch_size, num_classes)
        """
        # Create padding mask
        mask = self.create_padding_mask(x)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Attention pooling
        attention_weights = F.softmax(self.final_attention(x).squeeze(-1), dim=1)
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        
        # Classification
        output = self.classifier(x)
        
        return output

    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.01):
        """
        Configure optimizer with weight decay
        """
        # Separate parameters that should have weight decay from those that shouldn't
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Training step
        
        Args:
            batch: Tuple of (input_ids, labels)
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction
        
        Args:
            x: Input tensor of token indices
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
        return probs

