import argparse
import os
import random
from glob import glob
from multiprocessing import Pool
from typing import Optional, List, Union

import numpy as np
from miditoolkit import MidiFile, Note, Lyric
from tqdm import tqdm

from pretrain.tokenizer import BaseTokenizer

phrase_span_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


def get_phrase_spans(midi: MidiFile, tokenizer: BaseTokenizer) -> np.ndarray:
    _, notes_seg_indices = tokenizer.get_segment_indices(midi)

    phrase_spans = []
    start = 0
    end = 0
    for seg in notes_seg_indices:
        end = seg + 1
        phrase_spans.append((start, end))
        start = end
    
    assert len(phrase_spans) == len(notes_seg_indices)
    return np.array(phrase_spans, dtype=phrase_span_record_dtype)

def get_ngrams(ngram_file: str, word_mapping: List[tuple], length: Optional[int], top_p: Optional[float]) -> np.ndarray:
    ngrams = np.load(ngram_file)

    # filter ngrams
    if length is not None:
        ngrams = ngrams[ngrams["length"] <= length]
    if top_p is not None:
        ngrams = ngrams[ngrams["rank"] <= top_p]

    # remove ngrams covered by any longer ngram (longest match first)
    ngrams = ngrams[np.argsort(ngrams["length"])[::-1]]
    ngram_mask = np.ones(len(ngrams), dtype=bool)
    for i in range(len(ngrams)):
        if not ngram_mask[i]:
            continue
        start, end = ngrams["start"][i], ngrams["end"][i]
        covered_by_current = (ngrams["start"] >= start) & (ngrams["end"] <= end)
        covered_by_current[i] = False
        ngram_mask[covered_by_current] = False
    
    ngrams = ngrams[ngram_mask]
    for idx in range(len(ngrams)):
        ngrams[idx]["start"] = word_mapping[ngrams[idx]["start"]][0]
        ngrams[idx]["end"] = word_mapping[ngrams[idx]["end"] - 1][1]

    return ngrams

def get_word_mapping(notes: List[Note], lyrics: List[Lyric]):
    mixed_tokens: List[Union[Note, Lyric]] = []
    mixed_tokens.extend(notes)
    for lyric in lyrics:
        for token_index, token in enumerate(mixed_tokens):
            if type(token) != Note:
                continue
            if token.start < lyric.time:
                continue
            else:
                mixed_tokens.insert(token_index, lyric)
                break

    mappings = []
    tmp = [0, -1]
    for token in mixed_tokens:
        if type(token) == Lyric:
            if tmp[1] != -1:
                mappings.append(tuple(tmp))
                tmp = [tmp[1], -1]
            continue
        tmp[1] = notes.index(token) + 1
    if tmp[1] != -1:
        mappings.append(tuple(tmp))

    assert len(mappings) == len(lyrics)
    
    return mappings
    

def prepare_data_job(
    midi_file: str,
    pitch_peak_vowel_ngram_file: Optional[str],
    rhythm_vowel_ngram_file: Optional[str],
    dest_path: str,
    tokenizer: BaseTokenizer,
    task_id: int,
    ngram_length: Optional[int],
    ngram_top_p: Optional[float],
):
    """Prepare data for a single midi file. Return the length of the encoded data."""
    midi = MidiFile(midi_file, charset='utf8')
    lyrics = midi.lyrics
    notes = midi.instruments[0].notes
    word_mapping = get_word_mapping(notes, lyrics)

    data, token_map = tokenizer.encode(midi, task_id=task_id)
    results = {"data": data, "token_map": token_map}

    results["lyrics_length"] = len(lyrics)

    results["phrase_spans"] = get_phrase_spans(midi, tokenizer)
    results["phrase_spans"]["start"] += len(lyrics)
    results["phrase_spans"]["end"] += len(lyrics)

    # filter ngrams by max length with top_k
    if pitch_peak_vowel_ngram_file:
        results["pitch_peak_vowel_ngrams"] = get_ngrams(pitch_peak_vowel_ngram_file, word_mapping, ngram_length, ngram_top_p)
        results["pitch_peak_vowel_ngrams"]["start"] += len(lyrics)
        results["pitch_peak_vowel_ngrams"]["end"] += len(lyrics)
    if rhythm_vowel_ngram_file:
        results["rhythm_vowel_ngrams"] = get_ngrams(rhythm_vowel_ngram_file, word_mapping, ngram_length, ngram_top_p)
        results["rhythm_vowel_ngrams"]["start"] += len(lyrics)
        results["rhythm_vowel_ngrams"]["end"] += len(lyrics)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.savez(dest_path, **results)

    length, _ = data.shape
    return length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--kind", type=str, required=True)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--max_bar", type=int, default=128)
    parser.add_argument("--pitch_range", type=int, nargs=2, default=(0, 128))
    parser.add_argument("--velocity_range", type=int, nargs=2, default=(0, 128))
    parser.add_argument("--tempo_range", type=int, nargs=2, default=(30, 201))
    parser.add_argument("--ngram_length", type=int)
    parser.add_argument("--ngram_top_p", type=float)
    parser.add_argument("--total_task_num", type=int, default=1)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--pitch_peak_vowel_ngram_dir", type=str)
    parser.add_argument("--rhythm_vowel_ngram_dir", type=str)
        
    args = parser.parse_args()

    # prepare file args for each midi file
    print("finding files...")
    midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
    file_args = []
    for midi_file in midi_files:
        relpath = os.path.relpath(midi_file, args.midi_dir)
        basename = os.path.basename(midi_file)
        dest_path = os.path.join(args.dataset_dir, relpath.replace(".mid", ".npz"))
        pitch_peak_vowel_ngram_file, rhythm_vowel_ngram_file = None, None
        if args.pitch_peak_vowel_ngram_dir:
            pitch_peak_vowel_ngram_file = os.path.join(args.pitch_peak_vowel_ngram_dir, basename.replace(".mid", ".npy"))
        if args.rhythm_vowel_ngram_dir:
            rhythm_vowel_ngram_file = os.path.join(args.rhythm_vowel_ngram_dir, basename.replace(".mid", ".npy"))
        file_args.append((midi_file, pitch_peak_vowel_ngram_file, rhythm_vowel_ngram_file, dest_path))


    # prepare tokenizer
    tokenizer = BaseTokenizer.from_kwargs(
        metadata_dir=args.metadata_dir,
        kind=args.kind,
        granularity=args.granularity,
        max_bar=args.max_bar,
        pitch_range=args.pitch_range,
        velocity_range=args.velocity_range,
        tempo_range=args.tempo_range,
        total_task_num=args.total_task_num
    )

    # save tokenizer config for later use
    config_path = os.path.join(args.config_dir, "tokenizer_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    tokenizer.save_config(config_path)

    print(f"preparing {len(midi_files)} midi files...")
    with Pool() as pool:
        futures = [
            pool.apply_async(
                prepare_data_job,
                args=(*file_arg, tokenizer, args.task_id, args.ngram_length, args.ngram_top_p),
            )
            for file_arg in file_args
        ]
        lengths = [future.get() for future in tqdm(futures)]

    print(f"average data length: {np.mean(lengths)}")
