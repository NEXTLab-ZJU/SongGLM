import math
from typing import Dict, List
import torch
import torch.nn as nn
from .tokenizer import BaseTokenizer
from .dataset import positional_id_padding_index

class CompoundTokenFuser(nn.Module):
    """Fuses multiple token embeddings into a single embedding."""

    def __init__(self, tokenizer: BaseTokenizer, embedding_dim: Dict[str, int], model_dim: int, cutoffs: List[int], use_adaptive_embedding: bool = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)
        self.field_sizes = tokenizer.field_sizes
        self.total_field_size = sum(self.field_sizes)

        self.model_dim = model_dim
        self.total_embedding_dim = sum(embedding_dim[field_name] for field_name in tokenizer.field_names)
        
        # embedding size: embedding_dim[field_name]
        if use_adaptive_embedding:
            self.embeddings = nn.ModuleList()
            for field_name, field_size, pad_token_id in zip(tokenizer.field_names, tokenizer.field_sizes, tokenizer.pad_token_ids):
                if field_name != "lyrics":
                    self.embeddings.append(
                        nn.Embedding(
                            num_embeddings=field_size, embedding_dim=embedding_dim[field_name], padding_idx=pad_token_id
                        )
                    )
                else:
                    self.embeddings.append(
                        AdaptiveEmbedding(
                            num_embeddings=field_size, embedding_dim=embedding_dim[field_name], cutoffs=cutoffs, padding_idx=pad_token_id
                        )
                    )
        else:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=field_size, embedding_dim=embedding_dim[field_name], padding_idx=pad_token_id
                    )
                    for field_name, field_size, pad_token_id in zip(
                        tokenizer.field_names, tokenizer.field_sizes, tokenizer.pad_token_ids
                    )
                ]
            )
        self.encoder = nn.Linear(self.total_embedding_dim, model_dim)
        self.decoder = nn.Linear(model_dim, self.total_field_size)

    # encode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
        Returns:
            fused: (batch_size, seq_len, model_dim)
        """
        _, _, num_features = x.shape
        assert num_features == self.num_features, f"num_features must be {self.num_features}"

        # embeddings: (batch_size, seq_len, total_embedding_dim)
        x = torch.concat([embedding(x[:, :, i]) for i, embedding in enumerate(self.embeddings)], dim=2)
        # fused: (batch_size, seq_len, model_dim)
        x = self.encoder(x)
        return x

    def decode(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            fused: (batch_size, seq_len, model_dim)
        Returns:
            logits: List[torch.Tensor] of length num_features,
            each of shape (batch_size, seq_len, vocab_size of the feature)
        """
        # embeddings: (batch_size, seq_len, total_field_size)
        embeddings = self.decoder(fused)
        return torch.split(embeddings, self.field_sizes, dim=2)
    

class AdaptiveEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, cutoffs, padding_idx, div_value=4.):
        super(AdaptiveEmbedding, self).__init__()
        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= num_embeddings) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and num_embeddings-1")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cutoffs = cutoffs
        self.div_value = div_value
        self.padding_idx = padding_idx

        self.n_clusters = len(self.cutoffs) + 1
        self.edges = [0] + self.cutoffs + [num_embeddings]
        self.projections = nn.ModuleList()
        for i in range(self.n_clusters):
            hsz = int(self.embedding_dim // (self.div_value ** i))
            vsz = self.edges[i + 1] - self.edges[i]
            if padding_idx >= self.edges[i] and padding_idx <= self.edges[i + 1]:
                projection = nn.Sequential(
                    nn.Embedding(vsz, hsz, padding_idx=padding_idx - self.edges[i]),
                    nn.Linear(hsz, self.embedding_dim, bias=False)
                )
            else:
                projection = nn.Sequential(
                    nn.Embedding(vsz, hsz),
                    nn.Linear(hsz, self.embedding_dim, bias=False)
                )
            self.projections.append(projection)

    def forward(self, emb_input: torch.Tensor):

        batch_size, seq_len = emb_input.size()
        emb_input = emb_input.reshape(-1)
        emb_output = emb_input.new_empty(batch_size * seq_len, self.embedding_dim).half()

        for i in range(self.n_clusters):
            low_idx = self.edges[i]
            high_idx = self.edges[i + 1]
            input_mask = (emb_input >= low_idx) & (emb_input < high_idx)
            row_indices = input_mask.nonzero().squeeze(dim=-1)
            if row_indices.numel() == 0:
                continue
            input_subset = emb_input.index_select(0, row_indices)
            input_subset = input_subset - low_idx
            cluster_output = self.projections[i](input_subset)
            emb_output.index_copy_(0, row_indices, cluster_output)

        return emb_output.view(batch_size, seq_len, -1)


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer."""
    
    def __init__(self, model_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("positional_encoding", self._get_positional_encoding())

    # TODO: sinusoidal positional encoding, maybe try rotary embeddings!
    def _get_positional_encoding(self) -> torch.Tensor:
        positional_encoding = torch.zeros(self.max_seq_len, self.model_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-math.log(10000.0) / self.model_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch_size, seq_len, model_dim)
        Returns:
            x: (batch_size, seq_len, model_dim)
        """
        _, seq_len, _ = x.shape
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        return x
    
class CustomPositionalEncoding(nn.Module):
    """Custom Learnable Positional Encoding for the transformer, using provided positonal ids."""

    def __init__(self, model_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len + 1, model_dim, padding_idx=positional_id_padding_index)

    def forward(self, x: torch.Tensor, positional_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch_size, seq_len, model_dim)
            positional_ids: (batch_size, seq_len)
        Returns:
            x: (batch_size, seq_len, model_dim)
        """
        return x + self.embedding(positional_ids)
