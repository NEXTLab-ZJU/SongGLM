import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Type, Union
import pickle
import numpy as np
from miditoolkit import MidiFile, Note, Lyric
from tqdm import tqdm
from glob import glob
import pandas as pd
import re
from scipy.stats import entropy

from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.corpus import cmudict
from utils.analyze import note_analyze_per_midi
from utils.analyze import Note as FeatureNote


ticks_per_bar = 1920
Lexicon = Dict[Union[int, str], Dict[Tuple, float]]


def extract_ngrams(data, n):
    n_grams = ngrams(data, n)
    n_grams = list(n_grams)
    n_gram_indices = list(range(0, len(data) - n + 1))
    return list(zip(n_grams, n_gram_indices))

class NgramList:
    @staticmethod
    def from_raw(
        n: int,
        midi: MidiFile,
        PD: Optional[dict]
    ) -> List[Tuple]:
        raise NotImplementedError
    
    @staticmethod
    def split(
        n_gram: Tuple[Tuple]
    ) -> List[Tuple]:
        raise NotImplementedError


class NotePitchPeakNgramList(NgramList):
    @staticmethod
    def from_raw(n: int, midi: MidiFile, **kwargs) -> List[Tuple]:
        def pitch_feature_extract(note: FeatureNote):
            '''
            0 表示 non-melodic peak
            1 表示 melodic peak
            '''
            if note.is_peak:
                return 1
            elif note.is_non_peak:
                return 0
        
        lyrics = midi.lyrics
        midi_state = note_analyze_per_midi(midi)
        note_list = midi_state['note_list']
        assert len(note_list) == len(midi.instruments[0].notes)

        mixed_tokens: List[Union[FeatureNote, Lyric]] = []
        mixed_tokens.extend(note_list)
        for lyric in lyrics:
            for token_index, token in enumerate(mixed_tokens):
                if type(token) != FeatureNote:
                    continue
                if token.start < lyric.time:
                    continue
                else:
                    mixed_tokens.insert(token_index, lyric)
                    break
        
        pitch_note_groups = []
        note_pitch_feature_group = []
        for idx, token in enumerate(mixed_tokens):
            if type(token) == Lyric and idx != 0:
                pitch_note_groups.append(tuple(note_pitch_feature_group))
                note_pitch_feature_group = []
            elif idx != 0:
                note_pitch_feature_group.append(pitch_feature_extract(token))
        if len(note_pitch_feature_group) != 0:
            pitch_note_groups.append(tuple(note_pitch_feature_group))
        
        assert len(pitch_note_groups) == len(lyrics)

        return extract_ngrams(pitch_note_groups, n)

    @staticmethod
    def split(n_gram: Tuple[Tuple]) -> List[Tuple]:
        units = []
        for i in range(0, len(n_gram)):
            unit = (n_gram[i])
            units.append(unit)
        return units
    

class NoteRhythmNgramList(NgramList):
    @staticmethod
    def from_raw(n: int, midi: MidiFile, **kwargs) -> List[Tuple]:
        def time_features_extract(note: FeatureNote):
            '''
            0 表示节拍重音(Downbeat)
            1 表示长音(Agogic)
            2 表示切分音(Syncopation)
            3 表示节拍重长音
            4 表示节拍长切分音
            5 表示普通音符(Normal Notes)
            '''
            if note.is_downbeat and note.is_long:
                return 3
            elif note.is_split and note.is_long:
                return 4
            elif note.is_long:
                return 1
            elif note.is_downbeat:
                return 0
            elif note.is_split:
                return 2
            else:
                return 5

        lyrics = midi.lyrics
        midi_state = note_analyze_per_midi(midi)
        note_list = midi_state['note_list']
        assert len(note_list) == len(midi.instruments[0].notes)

        mixed_tokens: List[Union[FeatureNote, Lyric]] = []
        mixed_tokens.extend(note_list)
        for lyric in lyrics:
            for token_index, token in enumerate(mixed_tokens):
                if type(token) != FeatureNote:
                    continue
                if token.start < lyric.time:
                    continue
                else:
                    mixed_tokens.insert(token_index, lyric)
                    break
        
        time_note_groups = []
        note_time_feature_group = []
        for idx, token in enumerate(mixed_tokens):
            if type(token) == Lyric and idx != 0:
                time_note_groups.append(tuple(note_time_feature_group))
                note_time_feature_group = []
            elif idx != 0:
                note_time_feature_group.append(time_features_extract(token))
        if len(note_time_feature_group) != 0:
            time_note_groups.append(tuple(note_time_feature_group))

        assert len(time_note_groups) == len(lyrics)

        return extract_ngrams(time_note_groups, n)
    
    @staticmethod
    def split(n_gram: Tuple[Tuple]) -> List[Tuple]:
        units = []
        for i in range(0, len(n_gram)):
            unit = (n_gram[i])
            units.append(unit)
        return units


class LyricVowelNgramList(NgramList):
    @staticmethod
    def from_raw(n: int, midi: MidiFile, PD: dict) -> List[Tuple]:
        def get_vowel_features(lyric: Lyric) -> Tuple:
            '''
            0 表示无重音(Unstressed)
            1 表示主重音(Primary Stress)
            2 表示次重音(Secondary Stress)
            '''
            def clean_text(text):
                cleaned_text = re.sub('[^a-zA-Z\']', '', text)
                return cleaned_text.lower()

            vowel_features = []

            lyric_text_wo_dot = clean_text(lyric.text)
            pronunciation = PD[lyric_text_wo_dot][0]
            for syllable in pronunciation:
                if re.search(r"\d",syllable):
                    feature = re.findall(r'\d+',syllable)[0]
                    vowel_features.append(int(feature))
            
            return tuple(vowel_features)
        
        lyrics = midi.lyrics
        lyrics_feature_list = [get_vowel_features(lyric) for lyric in lyrics]

        return extract_ngrams(lyrics_feature_list, n)

    @staticmethod
    def split(n_gram: Tuple[Tuple]) -> List[Tuple]:
        units = []
        for i in range(0, len(n_gram)):
            unit = (n_gram[i])
            units.append(unit)
        return units


class NgramExtractor:
    def __init__(self, n_range: Tuple[int, int], note_ngram_type: Type[NgramList], lyric_ngram_type: Type[NgramList]) -> None:
        self.n_range = range(n_range[0], n_range[1] + 1)
        self.note_ngram_type = note_ngram_type
        self.lyric_ngram_type = lyric_ngram_type

    def save_config(self, dest_path: str):
        config = {"n_range": (self.n_range.start, self.n_range.stop - 1), "note_ngram_type": self.note_ngram_type.__name__, "lyric_ngram_type": self.lyric_ngram_type.__name__}
        with open(dest_path, "w") as f:
            json.dump(config, f)

    @staticmethod
    def from_config(config_path: str) -> "NgramExtractor":
        with open(config_path, "r") as f:
            config = json.load(f)
        n_range = tuple(config["n_range"])
        note_ngram_type = globals()[config["note_ngram_type"]]
        lyric_ngram_type = globals()[config["lyric_ngram_type"]]
        return NgramExtractor(n_range, note_ngram_type, lyric_ngram_type)


    # --------------------------------------------------------------------
    # stage0: extract metadata
    # --------------------------------------------------------------------
    def get_metadata(self, midi_files: List[str], dest_dir: str):
        pitch_set = set()
        onset_set = set()
        tempo_set = set()
        duration_set = set()
        velocity_set = set()
        lyrics = []

        def clean_text(text):
            cleaned_text = re.sub('[^a-zA-Z]', '', text)
            return cleaned_text.lower()
        
        print(f"extracting metadata from {len(midi_files)} files to {dest_dir}...")
        for midi_file in tqdm(midi_files):
            midi = MidiFile(midi_file)
            assert len(midi.instruments) == 1
            notes = midi.instruments[0].notes
            
            pitch_set = pitch_set.union(set([int(note.pitch) for note in notes]))
            onset_set = onset_set.union(set([int(note.start) for note in notes]))
            velocity_set = velocity_set.union(set([int(note.velocity) for note in notes]))
            duration_set = duration_set.union(set([int(note.end - note.start) for note in notes]))
            lyrics.extend([clean_text(lyric.text) for lyric in midi.lyrics])
            tempo_set = tempo_set.union(set([int(tempo_change.tempo) for tempo_change in midi.tempo_changes]))

        print("calculating word frequency...")
        lyrics_freq = FreqDist(lyrics)
        sorted_freq_dist = sorted(lyrics_freq.items(), key=lambda x: x[1], reverse=True)
        lyrics = [word for word, _ in sorted_freq_dist]

        metadata_json = {
            "pitch": list(pitch_set),
            "onset": list(onset_set),
            "tempo": list(tempo_set),
            "duration": list(duration_set),
            "velocity": list(velocity_set),
            "lyrics": list(lyrics)
        }
        
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        with open(dest_dir, "w") as f:
            json.dump(
                metadata_json,
                f,
                indent=4,
            )
    
    
    # --------------------------------------------------------------------
    # stage1: extract ngrams
    # --------------------------------------------------------------------
    def get_ngrams(self, midi_file: str) -> List[Tuple[Tuple, int]]:
        midi = MidiFile(midi_file)
        assert len(midi.instruments) == 1

        PD = cmudict.dict()
        
        note_ngram_lists = []
        lyric_ngram_lists = []

        for n in range(self.n_range.start, self.n_range.stop):
            note_ngram_list = self.note_ngram_type.from_raw(
                n,
                midi
            )

            lyric_ngram_list = self.lyric_ngram_type.from_raw(
                n,
                midi,
                PD
            )

            assert len(note_ngram_list) == len(lyric_ngram_list), f"Unmatched mapping! {(note_ngram_list)} {(lyric_ngram_list)}"

            note_ngram_lists.extend(note_ngram_list)
            lyric_ngram_lists.extend(lyric_ngram_list)

        return (note_ngram_lists, lyric_ngram_lists)
    
    def extract_ngrams_file(self, midi_file: str, dest_path: Optional[str] = None):
        ngrams = self.get_ngrams(midi_file)
        if dest_path:
            with open(dest_path, "wb") as f:
                pickle.dump(ngrams, f)
        else:
            note_ngram_lists, lyric_ngram_lists = ngrams
            print(note_ngram_lists)
            print(lyric_ngram_lists)

    def extract_ngrams(self, midi_files: List[str], dest_dir: str):
        print(f"extracting ngrams from {len(midi_files)} files to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        dest_paths = [
            os.path.join(dest_dir, os.path.basename(midi_file).replace(".mid", ".pkl")) for midi_file in midi_files
        ]
        with Pool() as pool:
            futures = [
                pool.apply_async(self.extract_ngrams_file, (midi_file, dest_path))
                for midi_file, dest_path in zip(midi_files, dest_paths)
            ]
            _ = [future.get() for future in tqdm(futures)]

    
    # --------------------------------------------------------------------
    # stage2: build lexicon
    # --------------------------------------------------------------------
    def get_ngram_frequency(self, ngram_file: str):
        with open(ngram_file, "rb") as f:
            note_ngram_lists, lyric_ngram_lists = pickle.load(f)
        note_frequency = defaultdict(int)
        note_frequency_by_length = defaultdict(int)
        lyric_frequency = defaultdict(int)
        lyric_frequency_by_length = defaultdict(int)

        note_lyric_distribution = {}
        
        for note_ngram_zip, lyric_ngram_zip in zip(note_ngram_lists, lyric_ngram_lists):
            note_ngram, _ = note_ngram_zip
            lyric_ngram, _ = lyric_ngram_zip

            note_frequency[note_ngram] += 1
            note_frequency_by_length[len(note_ngram)] += 1
            lyric_frequency[lyric_ngram] += 1
            lyric_frequency_by_length[len(lyric_ngram)] += 1

            if note_ngram not in note_lyric_distribution.keys():
                note_lyric_distribution[note_ngram] = {
                    lyric_ngram: 1
                }
            else:
                if lyric_ngram not in note_lyric_distribution[note_ngram].keys():
                    note_lyric_distribution[note_ngram][lyric_ngram] = 1
                else:
                    note_lyric_distribution[note_ngram][lyric_ngram] += 1

        return note_frequency, note_frequency_by_length, lyric_frequency, lyric_frequency_by_length, note_lyric_distribution
    
    def get_ngram_scores(
        self,
        type: str,
        prob: Dict[Tuple, float],
        count_by_length: Dict[int, float],
    ) -> Lexicon:
        """
        t-statistic score:
            - the higher the t-statistic score, the more likely it is a semantically-complete n-gram.
            - reference: ERNIE-GEN https://arxiv.org/abs/2001.11314
        """
        assert type in ["note", "lyric"], "Unknown type!"

        def _get_score(ngram: Tuple) -> float:
            if len(ngram) == 1:
                return 0

            iid_product = 1
            units = self.note_ngram_type.split(ngram) if type == "note" else self.lyric_ngram_type.split(ngram)
            for unit in units:
                iid_product *= prob[unit]
            product = prob[ngram]
            sigma_sqr = product * (1 - product)
            total_count = count_by_length[len(ngram)]
            t_statistic_score = (product - iid_product) / math.sqrt(sigma_sqr / total_count)
            return t_statistic_score

        scores: Lexicon = defaultdict(dict)
        for ngram in tqdm(prob.copy()):
            scores[len(ngram)][ngram] = _get_score(ngram)
        return scores
    
    def get_concentration_ratio(self, data, alpha = 1.0):
        base = 2
        assert len(data) >= 0
        if len(data) == 1:
            return 1
        arr = np.array(data)
        E = entropy(arr, base=base)
        NE = E / np.log2(len(data))
        concentration_ratio = 1 - alpha * NE
        return concentration_ratio

    def final_score(self, self_score, concentration_ratio, related_lyrics_scores, beta = 0.5):
        score = beta * (self_score) + (1 - beta) * concentration_ratio * np.mean(related_lyrics_scores)
        return score

    def build_lexicon(self, ngram_files: List[str], dest_path: str, ngram_type: str):
        print(f"loading ngrams from {len(ngram_files)} files...")
        note_frequency = defaultdict(int)
        note_frequency_by_length = defaultdict(int)
        lyric_frequency = defaultdict(int)
        lyric_frequency_by_length = defaultdict(int)
        note_lyric_distribution = {}

        with Pool(int(os.getenv('N_PROC', os.cpu_count()))) as pool:
            futures = [pool.apply_async(self.get_ngram_frequency, (ngram_file,)) for ngram_file in ngram_files]
            for future in tqdm(futures):
                note_freq, note_freq_by_length, lyric_freq, lyric_freq_by_length, note_lyric_dist = future.get()
                for ngram, count in note_freq.items():
                    note_frequency[ngram] += count
                for length, count in note_freq_by_length.items():
                    note_frequency_by_length[length] += count
                for ngram, count in lyric_freq.items():
                    lyric_frequency[ngram] += count
                for length, count in lyric_freq_by_length.items():
                    lyric_frequency_by_length[length] += count
                for note_ngram in note_lyric_dist.keys():
                    if note_ngram not in note_lyric_distribution.keys():
                        note_lyric_distribution[note_ngram] = note_lyric_dist[note_ngram]
                    else:
                        for lyric_ngram in note_lyric_dist[note_ngram].keys():
                            if lyric_ngram not in note_lyric_distribution[note_ngram].keys():
                                note_lyric_distribution[note_ngram][lyric_ngram] = note_lyric_dist[note_ngram][lyric_ngram]
                            else:
                                note_lyric_distribution[note_ngram][lyric_ngram] += note_lyric_dist[note_ngram][lyric_ngram]

        for note_ngram in note_lyric_distribution.keys():
            total = sum(list(note_lyric_distribution[note_ngram].values()))
            for lyric_ngram in note_lyric_distribution[note_ngram].keys():
                note_lyric_distribution[note_ngram][lyric_ngram] /= total

        print(f"total ngrams: {len(note_frequency)}")

        note_ngram_concentration_ratio = {}
        print("calculating concentration ratio ...")
        for note_ngram in tqdm(note_lyric_distribution.keys()):
            concentration_ratio = self.get_concentration_ratio(list(note_lyric_distribution[note_ngram].values()), alpha=1.0)
            note_ngram_concentration_ratio[note_ngram] = concentration_ratio
        
        print("calculating probs...")
        for ngram, count in tqdm(note_frequency.items()):
            note_frequency[ngram] = count / note_frequency_by_length[len(ngram)]
        for ngram, count in tqdm(lyric_frequency.items()):
            lyric_frequency[ngram] = count / lyric_frequency_by_length[len(ngram)]

        print("calculating scores...")
        note_lexicon = self.get_ngram_scores("note", note_frequency, note_frequency_by_length)
        lyric_lexicon = self.get_ngram_scores("lyric", lyric_frequency, lyric_frequency_by_length)
        
        # add lyric-related score
        print("adding lyric-related scores...")
        for ngram_len in tqdm(note_lexicon.keys()):
            for ngram in note_lexicon[ngram_len].keys():
                related_lyrics_scores = [lyric_lexicon[ngram_len][lyric_ngram] for lyric_ngram in note_lyric_distribution[ngram]]
                note_lexicon[ngram_len][ngram] = self.final_score(note_lexicon[ngram_len][ngram], note_ngram_concentration_ratio[ngram], related_lyrics_scores, beta=0.5)

        print("sorting scores by each length...")
        ngrams_csv = []
        for length, scores in note_lexicon.items():
            print(f"{length}-gram:", len(note_lexicon[length]))
            ngrams_csv.append([length, len(note_lexicon[length])])
            note_lexicon[length] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        df = pd.DataFrame(data=ngrams_csv, columns=['N', "Numbers"])
        df.to_csv(f"document/{ngram_type}.csv")

        print(f"saving to {dest_path}...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            pickle.dump(note_lexicon, f)

     
    # --------------------------------------------------------------------
    # stage3: rank ngrams
    # --------------------------------------------------------------------
    def get_lexicon_rank(self, lexicon: Lexicon) -> Dict[Tuple, float]:
        rank: Dict[Tuple, float] = {}
        for scores in lexicon.values():
            count = len(scores)
            for i, ngram in enumerate(scores):
                rank[ngram] = (i+1) / count
                # print(f"i = {i}, ngram = {ngram}, count = {count}, rank = {rank[ngram]}")
            
        return rank

    def get_ngram_labels(self, ngram_file: str, rank: Dict[Tuple, float]) -> np.ndarray:
        with open(ngram_file, "rb") as f:
            note_ngram_lists, _ = pickle.load(f)

        data = []
        for ngram, index in note_ngram_lists:
            if len(ngram) in self.n_range and ngram in rank.keys():
                data.append((index, index + len(ngram), len(ngram), rank[ngram]))
        
        # start word / end word
        result = np.array(
            data, dtype=[("start", np.int16), ("end", np.int16), ("length", np.int16), ("rank", np.float32)]
        )
        result.sort(order=["start", "length", "rank"])
        return result

    def prepare_ngram_labels(self, ngram_files: List[str], dest_dir: str, lexicon_path: str):
        print(f"loading lexicon from {lexicon_path}...")
        with open(lexicon_path, "rb") as f:
            lexicon: Lexicon = pickle.load(f)

        print(f"calculating lexicon rank...")
        lexicon = self.get_lexicon_rank(lexicon)

        print(f"preparing ngram labels for {len(ngram_files)} files to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        dest_paths = [
            os.path.join(dest_dir, os.path.basename(ngram_file).replace(".pkl", ".npy")) for ngram_file in ngram_files
        ]

        for ngram_file, dest_path in tqdm(zip(ngram_files, dest_paths), total=len(ngram_files)):
            ngram_labels = self.get_ngram_labels(ngram_file, lexicon)
            np.save(dest_path, ngram_labels)

