from typing import List, Optional, Union

import torch

from .dataset import (
    DataBatch,
    DataCollator,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForInfilling,
    DataCollatorForPaddingOnly,
    DataCollatorForGeneration,
    RandomPhraseMasking,
    RandomSpanMasking,
    RandomNgramMasking,
    SingleSpanMasking,
    MultiTargetInfillingMasking
)


class TrainingTask:
    def __init__(self, task_name: str, weight: float = 1.0):
        self.task_name = task_name
        self.weight = weight

    def get_data_collator(self) -> DataCollator:
        raise NotImplementedError

    def register_extra_modules(self, model) -> None:
        pass

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LanguageModelingTask(TrainingTask):
    """ causal language modeling """
    def __init__(
        self,
        seq_len: int,
        task_name: str = "clm",
        weight: float = 1.0,
        random_crop: bool = True,
        padding_only: bool = False,
    ):
        super().__init__(task_name, weight)
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.padding_only = padding_only

    def get_data_collator(self) -> DataCollator:
        if self.padding_only:
            return DataCollatorForPaddingOnly(seq_len=self.seq_len)
        return DataCollatorForCausalLanguageModeling(seq_len=self.seq_len, random_crop=self.random_crop)
    
    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


class InfillingTask(TrainingTask):
    def __init__(
        self,
        seq_len: int,
        task_name: str = "infilling",
        kind: Union[str, List[str]] = "span",
        weight: float = 1.0,
        probabilities: Optional[List[float]] = None,
        corruption_rate: float = 0.15,
        mean_span_length: int = 5,
        random_crop: bool = False,
        permutated_infilling: bool = False,
    ):
        super().__init__(f"{kind}_{task_name}", weight)
        self.kinds = kind if isinstance(kind, list) else [kind]
        self.probabilities = probabilities
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.permutated_infilling = permutated_infilling

    def get_data_collator(self) -> DataCollator:
        masking = get_masking(
            kinds=self.kinds,
            corruption_rate=self.corruption_rate,
            mean_span_length=self.mean_span_length,
            random_crop=self.random_crop,
            probabilities=self.probabilities
        )
        
        return DataCollatorForInfilling(
            masking=masking,
            seq_len=self.seq_len,
            random_crop=self.random_crop,
            permutated_infilling=self.permutated_infilling,
        )
    
    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


class GenerationTask:
    def __init__(
        self, 
        seq_len: int,
        task_name: str = "generation", 
        weight: float = 1.0
    ):
        self.task_name = task_name
        self.weight = weight
        self.seq_len = seq_len

    def get_data_collator(self) -> DataCollator:
        return DataCollatorForGeneration(seq_len=self.seq_len)

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


def get_masking(
    kinds: Union[str, List[str]],
    corruption_rate: float,
    mean_span_length: int,
    random_crop: bool,
    probabilities: Optional[List[float]] = None,
):
    def _get_masking(kind: str):
        if kind == "span":
            return RandomSpanMasking(corruption_rate=corruption_rate, mean_span_length=mean_span_length)
        elif kind == "pitch_peak_vowel_ngram":
            return RandomNgramMasking(
                corruption_rate=corruption_rate,
                fallback_mean_span_length=mean_span_length,
                extra_data_field_name="pitch_peak_vowel_ngrams",
            )
        elif kind == "rhythm_vowel_ngram":
            return RandomNgramMasking(
                corruption_rate=corruption_rate,
                fallback_mean_span_length=mean_span_length,
                extra_data_field_name="rhythm_vowel_ngrams",
            )
        elif kind == "phrase":
            return RandomPhraseMasking(corruption_rate=corruption_rate)
        elif kind == "single":
            return SingleSpanMasking(corruption_rate=corruption_rate)
        else:
            raise ValueError(f"Unknown infilling kind: {kind}")

    if len(kinds) == 1:
        return _get_masking(kinds[0])
    else:
        assert probabilities is None or len(probabilities) == len(kinds)
        if probabilities is None:
            probabilities = [1 / len(kinds)] * len(kinds)
        return MultiTargetInfillingMasking([_get_masking(kind) for kind in kinds], probabilities=probabilities)
