import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter

from tqdm import tqdm
from .dataset import DataBatch
from .module import CompoundTokenFuser, CustomPositionalEncoding, PositionalEncoding
from .task import TrainingTask
from .tokenizer import BaseTokenizer
from .utils import get_sampling_strategy, top_k_top_p_sample


DataBatchDict = Dict[str, DataBatch]


class Lyrics2MelodyModel(pl.LightningModule):
    """Base model for core step logic."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        use_adaptive_embedding: bool,
        cutoffs: List[int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_positional_encoding: bool = False,
        use_custom_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = BaseTokenizer.from_config(tokenizer_config_path)
        print(self.tokenizer)

        self.num_features = len(self.tokenizer.field_names)
        self.lyrics_index = self.tokenizer.field_names.index("lyrics")

        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.default_seq_len = 768
        self.prefix_source_length = self.default_seq_len
        self.prefix_length = 3 * self.default_seq_len

        self.fuser = CompoundTokenFuser(self.tokenizer, embedding_dim, model_dim, cutoffs, use_adaptive_embedding)

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(model_dim)

        self.use_custom_positional_encoding = use_custom_positional_encoding
        if use_custom_positional_encoding:
            self.custom_positional_encoding = CustomPositionalEncoding(model_dim, max_seq_len=self.default_seq_len)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    # triangular upper matrix
    def _create_causal_attention_mask(self, length: int) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=self.device), diagonal=1)
    
    def _get_causal_attention_mask(self, length: int) -> torch.Tensor:
        if not hasattr(self, "causal_attention_mask") or self.causal_attention_mask.shape[0] < length:
            self.causal_attention_mask = self._create_causal_attention_mask(max(length, self.default_seq_len))
        return self.causal_attention_mask[:length, :length]

    # attention mask in glm
    def _create_prefix_attention_mask(self, source_length: int, length: int) -> torch.Tensor:
        # bidirectional attention mask for prefix sequence
        left_prefix_part = torch.zeros((length, source_length), dtype=torch.bool, device=self.device)

        target_length = length - source_length
        top_right_target_part = torch.ones((source_length, target_length), dtype=torch.bool, device=self.device)

        # causal attention mask for infilling sequence
        bottom_right_target_part = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=self.device), diagonal=1
        )

        right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
        return torch.cat([left_prefix_part, right_target_part], dim=1)
    
    def _get_prefix_attention_mask(self, source_length: int, length: int) -> torch.Tensor:
        # return self._create_prefix_attention_mask(source_length, length)
        if (
            not hasattr(self, "prefix_attention_mask")
            or self.prefix_source_length < source_length
            or self.prefix_length < length
            or length - source_length > self.prefix_length - self.prefix_source_length
        ):
            new_source_length = max(source_length, self.prefix_source_length)
            new_length = new_source_length + max(length - source_length, self.prefix_length - self.prefix_source_length)
            self.prefix_attention_mask = self._create_prefix_attention_mask(new_source_length, new_length)
            self.prefix_source_length = new_source_length
            self.prefix_length = new_length
        start = self.prefix_source_length - source_length
        end = start + length
        assert end <= self.prefix_length
        return self.prefix_attention_mask[start:end, start:end]

    
    def _get_padding_mask(self, lengths: List[int], seq_len: int) -> torch.Tensor:
        padding_mask = torch.zeros((len(lengths), seq_len), dtype=torch.bool, device=self.device)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return padding_mask

    def forward(self, batch: DataBatch, return_outputs: bool = False) -> List[torch.Tensor]:
        """Args:
            batch: DatasetBatch
            return_outputs: bool, whether to return transformer encoder outputs
        Returns:
            decoded: list of num_features * (batch_size, seq_len, vocab_size of the feature)
            outputs: (batch_size, seq_len, model_dim), transformer encoder outputs
        """
        batch_size, seq_len, _ = batch.input_ids.shape

        padding_mask = self._get_padding_mask(batch.lengths, seq_len)
        if batch.attention_kind == "full":
            attention_mask = None
        elif batch.attention_kind == "causal":
            attention_mask = self._get_causal_attention_mask(seq_len)
        elif batch.attention_kind == "prefix":
            attention_mask = (
                torch.stack(
                    [self._get_prefix_attention_mask(source_length, seq_len) for source_length in batch.source_lengths]
                )
                .view(batch_size, 1, seq_len, seq_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(batch_size * self.num_heads, seq_len, seq_len)
            )

        x = self.fuser(batch.input_ids)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        if self.use_custom_positional_encoding and batch.positional_ids is not None:
            x = self.custom_positional_encoding(x, batch.positional_ids)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask, mask=attention_mask)
        decoded = self.fuser.decode(x)

        if return_outputs:
            return decoded, x
        else:
            return decoded

    def _get_loss(
        self, logits: List[torch.Tensor], label_ids: torch.Tensor, ignore_lyrics_loss: bool = True, return_parts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        losses = []
        for i, logit in enumerate(logits):
            # ignore lyrics loss when predicting notes
            if ignore_lyrics_loss:
                if i == self.lyrics_index:
                    continue
            
            # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
            loss = F.cross_entropy(
                logit.transpose(1, 2), label_ids[:, :, i], ignore_index=self.tokenizer.pad_token_ids[i]
            )
            losses.append(loss)
            self.log(f"{self.tokenizer.field_names[i]}_loss", loss, sync_dist=True)
        loss = torch.stack(losses).mean()
        if return_parts:
            return loss, losses
        return loss


class Lyrics2MelodyPretrainModel(Lyrics2MelodyModel):
    """Use this subclass for pretraining or finetuning the model."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        cutoffs: List[int],
        use_adaptive_embedding: bool,
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Optimizer hyperparameters
        lr: float,
        betas: Tuple[float, float],
        epsilon: float,
        weight_decay: float,
        warmup_percent: float,
        fixed_lr: bool = False,
        # Training configuration
        use_positional_encoding: bool = False,
        use_custom_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            cutoffs=cutoffs,
            use_adaptive_embedding=use_adaptive_embedding,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_custom_positional_encoding=use_custom_positional_encoding,
        )

        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.warmup_percent = warmup_percent
        self.fixed_lr = fixed_lr

        self.tasks: Dict[str, TrainingTask] = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.betas, eps=self.epsilon, weight_decay=self.weight_decay
        )
        if self.fixed_lr:
            return optimizer
        else:
            total_steps = (
                self.trainer.max_steps if self.trainer.max_steps > 0 else self.trainer.estimated_stepping_batches
            )
            print(f"Total steps: {total_steps}")
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                anneal_strategy="cos",
                pct_start=self.warmup_percent,
            )
            scheduler = {"scheduler": lr_scheduler, "interval": "step"}
            return [optimizer], [scheduler]

    def register_task(self, task: TrainingTask):
        self.tasks[task.task_name] = task
        task.register_extra_modules(self)

    def _get_batch_size(self, batch: DataBatchDict) -> int:
        if len(self.tasks) == 1:
            return batch.input_ids.shape[0]
        return next(iter(batch.values())).input_ids.shape[0]
    
    def _shared_step(self, batch: DataBatchDict) -> torch.Tensor:
        if len(self.tasks) == 1:
            task = next(iter(self.tasks.values()))
            return task(self, batch), {}
        losses = {}
        for task_name, task in self.tasks.items():
            losses[task_name] = task(self, batch[task_name])
        weighted_losses = [loss * self.tasks[task_name].weight for task_name, loss in losses.items()]
        loss = torch.stack(weighted_losses).sum()
        return loss, losses
    
    def training_step(self, batch: DataBatchDict, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, any]]:
        batch_size = self._get_batch_size(batch)
        loss, losses = self._shared_step(batch)
        if len(self.tasks) > 1:
            for task_name in self.tasks:
                self.log(f"train_loss:{task_name}", losses[task_name], sync_dist=True, batch_size=batch_size)
        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch: DataBatchDict, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, any]]:
        batch_size = self._get_batch_size(batch)
        loss, _ = self._shared_step(batch)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size, prog_bar=True)
        return loss


class Lyrics2MelodyCompletionModel(Lyrics2MelodyModel):
    """Use this subclass for the melody completion downstream task."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        cutoffs: List[int],
        use_adaptive_embedding: bool,
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        advanced_sampling: bool = False,
        max_length: int = 768,
        times_to_predict: int = 1,
        use_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            cutoffs=cutoffs,
            use_adaptive_embedding=use_adaptive_embedding,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.times_to_predict = times_to_predict
        self.advanced_sampling = advanced_sampling

        self.lyrics_bos_token_tensor = torch.tensor(self.tokenizer.bos_token_ids, dtype=torch.long)
        self.lyrics_eos_token_tensor = torch.tensor(self.tokenizer.eos_token_ids, dtype=torch.long)
        self.note_bos_token_tensor = torch.tensor(self.tokenizer.bos_token_ids, dtype=torch.long)
        self.note_eos_token_tensor = torch.tensor(self.tokenizer.eos_token_ids, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> List[torch.Tensor]:
        x = self.fuser(input_ids)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask)
        return self.fuser.decode(x)

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids = batch.input_ids

        if self.lyrics_bos_token_tensor.device != self.device or self.lyrics_eos_token_tensor.device != self.device or self.note_bos_token_tensor.device != self.device or self.note_eos_token_tensor.device != self.device:
            self.lyrics_bos_token_tensor = self.lyrics_bos_token_tensor.to(self.device)
            self.lyrics_eos_token_tensor = self.lyrics_eos_token_tensor.to(self.device)
            self.note_bos_token_tensor = self.note_bos_token_tensor.to(self.device)
            self.note_eos_token_tensor = self.note_eos_token_tensor.to(self.device)

        batch_size, seq_len, _ = batch.input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        input_ids = input_ids[0]
        input_ids = input_ids[:batch.source_lengths[0]]
        input_ids = torch.cat((input_ids, self.note_bos_token_tensor.unsqueeze(0)), dim=0)

        input_ids = input_ids.expand(self.times_to_predict, input_ids.shape[0], -1)
        reached_eos = np.zeros(self.times_to_predict, dtype=np.bool_)

        attention_mask = self._get_prefix_attention_mask(batch.source_lengths[0], self.max_length)

        while input_ids.shape[1] < self.max_length:
            seq_len = input_ids.shape[1]
            attn_mask = attention_mask[:seq_len, :seq_len]
            logits = self(input_ids, attn_mask=attn_mask)
            
            sampled_tokens = []

            for field_index, logit in enumerate(logits):
                # Decode according to the sampling strategy
                if self.advanced_sampling:
                    sampled_token = top_k_top_p_sample(logit[:, -1, :], **get_sampling_strategy(field_index))
                else:
                    sampled_token = top_k_top_p_sample(
                        logit[:, -1, :], top_k=self.top_k, top_p=self.top_p, temperature=self.temperature
                    )
                sampled_tokens.append(sampled_token)
            # (batch_size, num_features)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            # until <EOS> token is generated or the predicted bar length is reached
            for batch_index in range(self.times_to_predict):
                token = sampled_tokens[batch_index]
                if torch.all(token == self.note_eos_token_tensor):
                    reached_eos[batch_index] = True
            if np.all(reached_eos):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens[:, None, :]], dim=1)
        
        return input_ids

class CustomWriter(BasePredictionWriter):
    """Write the prediction to a MIDI file."""

    def __init__(self, output_dir: str):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def decode(
        self, token_ids: np.ndarray, tokenizer: BaseTokenizer, prediction_bar_length: Optional[int] = None, **kwargs
    ):
        # (seq_len, num_features)
        # Crop the sequence at the first <EOS> token or where the bar length is reached
        midi = tokenizer.decode(token_ids)
        midi = tokenizer.filter_overlapping_notes(midi)
        return midi

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: Lyrics2MelodyCompletionModel,
        prediction: torch.Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: DataBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if prediction is None:
            return
        filename = batch.filenames[0] if batch.filenames is not None else str(batch_idx)

        # (batch_size, seq_len, num_features)
        prediction = prediction.cpu().numpy()
        for index in range(prediction.shape[0]):
            if prediction.shape[0] > 1:
                dest_path = os.path.join(self.output_dir, f"batch_{index}", f"{filename}.mid")
            else:
                dest_path = os.path.join(self.output_dir, f"{filename}.mid")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            midi = self.decode(
                prediction[index],
                tokenizer=pl_module.tokenizer,
                prediction_bar_length=pl_module.prediction_bar_length
                if hasattr(pl_module, "prediction_bar_length")
                else None,
            )
            try:
                midi.dump(dest_path)
            except Exception:
                continue
