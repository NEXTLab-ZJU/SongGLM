import re
import math
import json
import numpy as np
from math import floor, log2
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from miditoolkit import Instrument, MidiFile, Note, TempoChange, Lyric
from operator import attrgetter
from collections import defaultdict

AnyField = Union[int, str]
pad_str = "<PAD>"
token_map_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


class BaseTokenizer:
    kind: Optional[str] = None

    class Token(NamedTuple):
        """NamedTuple for MIDI and lyrics compound token.
        One compound token represents one note. Each field is a feature of the note."""

        pass

    def __init__(
        self,
        metadata_dir: str,
        granularity: int = 64,
        max_bar: int = 128,
        pitch_range: Tuple[int, int] = (0, 128),
        velocity_range: Tuple[int, int] = (0, 128),
        tempo_range: Tuple[int, int] = (30, 201),
        group_range: Tuple[int, int] = (0, 256),
        phrase_range: Tuple[int, int] = (0, 256),
        total_task_num: int = 1
    ) -> None:
        """Initialize a BaseTokenizer instance.
        Args:
            granularity: The number of units per bar. Defaults to 64 (64-th note).
            max_bar: The maximum number of bar token to use. Exceeded ones will be mod by the number. Defaults to 128.
            pitch_range: The range of pitch token to use. Defaults to (0, 128)."""
        
        self.granularity = granularity
        self.max_bar = max_bar
        self.metadata_dir = metadata_dir
        self.pitch_range = pitch_range
        self.velocity_range = velocity_range
        self.tempo_range = tempo_range
        self.group_range = group_range
        self.phrase_range = phrase_range
        self.total_task_num = total_task_num

        # define bins for each field
        self.ticks_per_bar = 1920
        self.units_per_bar = granularity
        self.ticks_per_beat = self.ticks_per_bar // 4
        self.ticks_per_unit = self.ticks_per_bar // self.units_per_bar
        self.ticks_per_triplet_unit = self.ticks_per_unit // 3 * 4

        self.define_bins()

        self.vocabularies: Dict[str, List[Union[int, str]]] = {}
        self.define_vocabularies()
        self.field_names = self.Token._fields
        self.field_indices = {name: index for index, name in enumerate(self.field_names)}
        self.vocab_sizes = [len(self.vocabularies[field_name]) for field_name in self.field_names]
        self.field_sizes = list(self.vocab_sizes)  # will be modified when adding special tokens

        self.define_special_tokens()
        self.build_encoder_decoder()
        self.set_special_token_ids()

    def define_bins(self) -> None:
        metadata = {}
        with open(self.metadata_dir, "r") as f:
            metadata = json.load(f)

        # pitch
        pitch = metadata["pitch"]
        self.pitch_range = range(self.pitch_range[0], self.pitch_range[1])
        assert all(pit in list(self.pitch_range) for pit in pitch)
        
        # position
        double_positions = set(range(0, self.ticks_per_bar, self.ticks_per_unit))
        triplet_positions = set(range(0, self.ticks_per_bar, self.ticks_per_triplet_unit))
        onset = [on % self.ticks_per_bar for on in metadata["onset"]]
        self.position_bins = sorted(double_positions | triplet_positions)
        assert all(on in self.position_bins for on in onset)

        # duration
        double_duration = set(range(self.ticks_per_unit, self.ticks_per_bar + 1, self.ticks_per_unit))
        triplet_ratio = floor(log2(self.granularity / 3))
        triplet_duration = set([self.ticks_per_bar // (3 * 2**r) for r in range(triplet_ratio + 1)])
        # duration = metadata["duration"]
        self.duration_bins = sorted(double_duration | triplet_duration)
        # assert all(dur in self.duration_bins for dur in duration)

        # tempo
        tempo = metadata["tempo"]
        self.tempo_range = range(self.tempo_range[0], self.tempo_range[1])
        self.default_tempo = 120
        # assert all(temp in list(self.tempo_range) + [4] for temp in tempo)

        # velocity
        velocity = metadata["velocity"]
        self.velocity_range = range(self.velocity_range[0], self.velocity_range[1])
        assert all(vel in list(self.velocity_range) for vel in velocity)

        # group
        self.group_range = range(self.group_range[0], self.group_range[1])

        # phrase
        self.phrase_range = range(self.phrase_range[0], self.phrase_range[1])

        # lyrics
        lyrics = metadata["lyrics"]
        self.lyrics_bins: List[str] = lyrics

        # token type
        self.token_type_bins = ["note", "lyrics"]

        # task id
        self.task_id_bins = range(self.total_task_num)

    def define_vocabularies(self) -> None:
        raise NotImplementedError
    
    def define_special_tokens(self) -> None:
        self.bos_token_str = "<BOS>"
        self.eos_token_str = "<EOS>"
        self.pad_token_str = "<PAD>"
        self.sep_token_str = "<SEP>"
        self.cls_token_str = "<CLS>"
        self.mask_token_str = "[MASK]"
        self.long_mask_token_str = "[lMASK]"

        self.bos_token = self.Token(*[self.bos_token_str] * len(self.field_names))
        self.eos_token = self.Token(*[self.eos_token_str] * len(self.field_names))
        self.pad_token = self.Token(*[self.pad_token_str] * len(self.field_names))
        self.sep_token = self.Token(*[self.sep_token_str] * len(self.field_names))
        self.cls_token = self.Token(*[self.cls_token_str] * len(self.field_names))
        self.mask_token = self.Token(*[self.mask_token_str] * len(self.field_names))
        self.long_mask_token = self.Token(*[self.long_mask_token_str] * len(self.field_names))

        self.special_token_str = [
            self.bos_token_str,
            self.eos_token_str,
            self.pad_token_str,
            self.sep_token_str,
            self.cls_token_str,
            self.mask_token_str,
            self.long_mask_token_str,
        ]

    def build_encoder_decoder(self) -> None:
        # token to id map
        self.encoder: Dict[str, Dict[AnyField, int]] = {
            field_name: {field: index for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }
        # id to token map
        self.decoder: Dict[str, Dict[int, AnyField]] = {
            field_name: {index: field for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }

        # add special tokens to the encoder and decoder
        # put special token at last
        for field_index, field_name in enumerate(self.field_names):
            for i, token_str in enumerate(self.special_token_str):
                token_id = len(self.vocabularies[field_name]) + i
                self.encoder[field_name][token_str] = token_id
                self.decoder[field_name][token_id] = token_str
            self.field_sizes[field_index] += len(self.special_token_str)

    def set_special_token_ids(self) -> None:
        self.bos_token_ids = self.convert_token_to_id(self.bos_token)
        self.eos_token_ids = self.convert_token_to_id(self.eos_token)
        self.pad_token_ids = self.convert_token_to_id(self.pad_token)
        self.sep_token_ids = self.convert_token_to_id(self.sep_token)
        self.cls_token_ids = self.convert_token_to_id(self.cls_token)
        self.mask_token_ids = self.convert_token_to_id(self.mask_token)
        self.long_mask_token_ids = self.convert_token_to_id(self.long_mask_token)

    def tokenize(self, midi: MidiFile, task_id: int = 0) -> Tuple[List[Token], np.ndarray]:
        """Returns:
        tokens: list of tokens.
        note_map: (num_note) with columns [start, end] mapping note index to token index."""
        raise NotImplementedError

    def detokenize(self, tokens: List[Token]) -> MidiFile:
        raise NotImplementedError
    
    def convert_tokens_to_ids(self, tokens: List[Token]) -> np.ndarray:
        token_ids = np.zeros((len(tokens), len(self.field_names)), dtype=np.int32)
        for index, token in enumerate(tokens):
            for field_index, field_name in enumerate(self.field_names):
                field = token[field_index]
                token_ids[index, field_index] = (
                    self.encoder[field_name][field] if field is not None else self.pad_token_ids[field_index]
                )
        return token_ids
    
    def convert_ids_to_tokens(self, tokens: np.ndarray) -> List[Token]:
        assert tokens.ndim == 2, "tokens should be 2D array."
        length, field_count = tokens.shape
        assert field_count == len(self.field_names), "field count should be equal to field names."

        result: List[self.Token] = []
        for index in range(length):
            fields = []
            for field_index, field_name in enumerate(self.field_names):
                token = tokens[index, field_index]
                field = self.decoder[field_name].get(token, None)
                fields.append(field)
            result.append(self.Token(*fields))
        return result
    
    def convert_token_to_id(self, token: Token) -> np.ndarray:
        return self.convert_tokens_to_ids([token])[0]

    def convert_id_to_token(self, token: np.ndarray) -> Token:
        return self.convert_ids_to_tokens(np.expand_dims(token, axis=0))[0]
    
    def encode(self, midi: MidiFile, task_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Encode midi file to token ids.
        Args:
            midi: midi file to encode.
        Returns:
            token_ids: (length, field).
            note_map: (num_note) with columns [start, end] mapping note index to token index.
        """
        tokens, note_map = self.tokenize(midi, task_id=task_id)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids, note_map

    def decode(self, token_ids: np.ndarray) -> MidiFile:
        """Decode token ids to midi file.
        Args:
            token_ids: (length, field)
        Returns:
            midi: decoded midi file.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.detokenize(tokens)
    
    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        """Pitch shift augumentation. This method will modify the token_ids in place.
        Args:
            token_ids: (num_tokens, num_fields)
            shift_range: pitch shift range in semitone. The direction may be upward or downward.
        """
        raise NotImplementedError

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        """Get the bar length of the tokens."""
        raise NotImplementedError

    def get_notes_seg_indices(self, midi: MidiFile) -> List[int]:
        """Get indices of the segment notes of the midi file.
        The segment notes are:
        (1) the longest notes in each bar (longer than quarter note),
        (2) notes following at least eighth rest."""

        ticks_per_beat = midi.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4

        def filter_consecutive_notes(note_indices: List[int], notes: List[Note]):
            if len(note_indices) <= 1:
                return note_indices

            note_indices = np.array(note_indices)
            consecutive_mask = (note_indices[1:] - note_indices[:-1]) == 1
            result = []
            index = 0
            while index < len(note_indices) - 1:
                note_index = note_indices[index]
                if not consecutive_mask[index]:
                    result.append(note_index)
                    if index == len(note_indices) - 2:
                        # add the last note
                        result.append(note_indices[-1])
                    index += 1
                else:
                    # two consecutive notes
                    cur_note = notes[note_index]
                    next_note = notes[note_indices[index + 1]]
                    distance = next_note.start - cur_note.end
                    # if the distance between two consecutive notes is greater than eighth rest,
                    # or the first note is longer than the second note, keep the first note;
                    # otherwise, keep the second note.
                    if distance >= ticks_per_beat // 2:
                        result.append(note_index)
                    elif cur_note.end - cur_note.start > next_note.end - next_note.start:
                        result.append(note_index)
                    else:
                        result.append(note_indices[index + 1])
                    index += 2
            return result

        notes = midi.instruments[0].notes

        # get the longest notes in each bar (longer than quarter note)
        bar_dict = defaultdict(list)
        for note in notes:
            bar_dict[note.start // ticks_per_bar].append(note)
        long_notes = []
        for bar_notes in bar_dict.values():
            max_duration = max(x.end - x.start for x in bar_notes)
            if max_duration > ticks_per_beat:
                long_notes.extend(note for note in bar_notes if note.end - note.start == max_duration)
        long_note_indices = sorted(notes.index(note) for note in long_notes)
        long_note_indices = filter_consecutive_notes(long_note_indices, notes)

        # get notes following at least eighth rest
        rest_note_indices = [
            i
            for i, note in enumerate(notes)
            if i == len(notes) - 1 or notes[i + 1].start - note.end >= ticks_per_beat // 2
        ]

        # combine long notes and rest notes, then filter out consecutive notes
        segment_note_indices = sorted(set(long_note_indices) | set(rest_note_indices))
        segment_note_indices = filter_consecutive_notes(segment_note_indices, notes)

        # if the first segment note is the first note, and is too close to the second note, remove the first note
        if (
            len(segment_note_indices) >= 2
            and segment_note_indices[0] == 0
            and notes[1].start - notes[0].end < ticks_per_beat // 2
        ):
            segment_note_indices = segment_note_indices[1:]
        
        if len(notes) - 1 not in segment_note_indices:
            segment_note_indices.append(len(notes) - 1)
        
        return segment_note_indices

    def get_lyrics_seg_indices(self, midi: MidiFile) -> List[int]:
        lyrics = midi.lyrics
        lyrics_text = [l.text for l in lyrics]

        seg_indices = []

        for idx, lyric in enumerate(lyrics_text):
            if lyric[-1] in list(".,!;:?"):
                seg_indices.append(idx)

        if len(lyrics) - 1 not in seg_indices:
            seg_indices.append(len(lyrics) - 1)

        return seg_indices

    def midi_note_lyric_tag(self, midi: MidiFile):
        mixed_tokens: List[Union[Note, Lyric]] = []
        mixed_tokens.extend(midi.instruments[0].notes)
        for lyric in midi.lyrics:
            for token_index, token in enumerate(mixed_tokens):
                if type(token) != Note:
                    continue
                if token.start < lyric.time:
                    continue
                else:
                    mixed_tokens.insert(token_index, lyric)
                    break

        lyric_index = -1
        for token in mixed_tokens:
            if type(token) == Lyric:
                lyric_index += 1
            else:
                note_index = midi.instruments[0].notes.index(token)
                midi.instruments[0].notes[note_index].tag = lyric_index
        
        return midi

    def get_phrase_seg(self, midi: MidiFile) -> Tuple[List[int], List[int]]:

        def get_punctuation_number(lyrics: List[Lyric]):
            lyrics_str = [l.text for l in lyrics]
            cnt = 0
            for lyric in lyrics_str:
                if lyric[-1] in list(".,!;:?"):
                    cnt += 1
            return cnt
        
        midi = self.midi_note_lyric_tag(midi)
        punctuation_number = get_punctuation_number(lyrics=midi.lyrics)
        
        # print(f"punctuation ratio: {punctuation_number / len(midi.lyrics)}")

        if punctuation_number / len(midi.lyrics) < 1/12:
            notes_seg_indices = self.get_notes_seg_indices(midi)
            lyrics_seg_indices = []
            for note_index in notes_seg_indices:
                lyrics_seg_indices.append(midi.instruments[0].notes[note_index].tag)
            
            if len(midi.lyrics) - 1 not in lyrics_seg_indices:
                lyrics_seg_indices.append(len(midi.lyrics) - 1)
            
            lyrics_seg_indices = list(set(lyrics_seg_indices))
            lyrics_seg_indices.sort()
        else:
            lyrics_seg_indices = self.get_lyrics_seg_indices(midi)
        
        notes_seg_indices = []
        for lyric_index in lyrics_seg_indices:
            lyric_end = midi.instruments[0].notes[-1].end if lyric_index == len(midi.lyrics) - 1 else midi.lyrics[lyric_index + 1].time
            for idx, note in enumerate(midi.instruments[0].notes):
                if note.end <= lyric_end:
                    continue
                else:
                    notes_seg_indices.append(idx - 1)
                    break
        if len(midi.instruments[0].notes) - 1 not in notes_seg_indices:
            notes_seg_indices.append(len(midi.instruments[0].notes) - 1)
        
        notes_seg_indices = list(set(notes_seg_indices))
        notes_seg_indices.sort()

        assert len(lyrics_seg_indices) == len(notes_seg_indices)
        return lyrics_seg_indices, notes_seg_indices

    def get_segment_indices(self, midi: MidiFile) -> Tuple[List[int], List[int]]:
        """Get indices of the segment notes/lyrics of the midi file.
        The segment notes are from MelodyGLM:
        (1) the longest notes in each bar (longer than quarter note),
        (2) notes following at least eighth rest.
        The segment lyrics are from punctuations: .!?;:
        """
        
        lyrics_seg_indices, notes_seg_indices = self.get_phrase_seg(midi)

        return lyrics_seg_indices, notes_seg_indices

    def _get_tempo_changes(self, midi: MidiFile) -> List[TempoChange]:
        # sort and deduplicate tempo changes
        tempo_changes = midi.tempo_changes
        if len(tempo_changes) == 0:
            tempo_changes = [TempoChange(tempo=self.default_tempo, time=0)]
        elif len(tempo_changes) > 1:
            tempo_changes = sorted(midi.tempo_changes, key=lambda x: x.time)
            tempo_changes = [
                tempo_changes[i]
                for i in range(len(tempo_changes))
                if i == len(tempo_changes) - 1 or tempo_changes[i].time != tempo_changes[i + 1].time
            ]
        return tempo_changes
    
    def filter_overlapping_notes(self, midi: MidiFile) -> MidiFile:
        assert len(midi.instruments) == 1
        # sort notes by start time, end time (longer first), and pitch (higher first)
        notes = sorted(midi.instruments[0].notes, key=lambda x: (x.start, -x.end, -x.pitch))
        new_notes = []
        current_end = None
        for note in notes:
            if current_end is None or note.end > current_end:
                # the note is not overlapped
                new_notes.append(note)
                current_end = note.end
        midi.instruments[0].notes = new_notes
        return midi

    def _find_nearest(self, bins: List[Union[int, str]], value: Union[int, str]) -> Union[int, str]:
        """Find the nearest bin to the value."""
        if type(value) == str:
            # assert value in bins, f"can't find exact value: {value}."
            if value in bins:
                return value
            else:
                return "<NONE>"
        elif type(value) == int:
            return min(bins, key=lambda x: abs(x - value) if type(x) == int else math.inf)
        else:
            # raise ValueError(f"{value} must be int or str.")
            return "<NONE>"
    
    def __str__(self) -> str:
        info_str = f"representation: {self.kind}, granularity={self.granularity}"
        token_size_str = ", ".join([f"{field_name}={len(d)}" for field_name, d in self.encoder.items()])
        return info_str + "\n" + token_size_str
    
    def save_config(self, path: str):
        if self.kind is None:
            raise NotImplementedError
        config = {
            "kind": self.kind,
            "metadata_dir": self.metadata_dir,
            "granularity": self.granularity,
            "max_bar": self.max_bar,
            "pitch_range": [self.pitch_range.start, self.pitch_range.stop],
            "velocity_range": [self.velocity_range.start, self.velocity_range.stop],
            "tempo_range": [self.tempo_range.start, self.tempo_range.stop],
            "group_range": [self.group_range.start, self.group_range.stop],
            "phrase_range": [self.phrase_range.start, self.phrase_range.stop],
            "total_task_num": self.total_task_num
        }
        with open(path, "w") as f:
            json.dump(config, f)

    @staticmethod
    def from_kwargs(kind: str, **kwargs) -> "BaseTokenizer":
        if kind == "octuple_like":
            return OctupleLikeTokenizer(**kwargs)
        # elif kind == "cp":
        #     return CPTokenizer(**kwargs)
        # elif kind == "remi":
        #     return RemiTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
    @staticmethod
    def from_config(path: str) -> "BaseTokenizer":
        with open(path) as f:
            config = json.load(f)
        kind = config.pop("kind")
        return BaseTokenizer.from_kwargs(kind, **config)


class OctupleLikeTokenizer(BaseTokenizer):
    kind = "octuple_like"

    class Token(NamedTuple):
        bar: AnyField
        position: AnyField
        
        duration: AnyField
        pitch: AnyField
        tempo: AnyField
        velocity: AnyField
        lyrics: AnyField

        group: AnyField
        phrase: AnyField
        token_type: AnyField
        task_id: AnyField

        def __str__(self) -> str:
            bar = self.bar if self.bar != pad_str else ""
            position = self.position if self.position != pad_str else ""
            duration = self.duration if self.duration != pad_str else ""
            pitch = self.pitch if self.pitch != pad_str else ""
            tempo = self.tempo if self.tempo != pad_str else ""
            velocity = self.velocity if self.velocity != pad_str else ""
            lyrics = self.lyrics if self.lyrics != pad_str else ""
            group = self.group if self.group != pad_str else ""
            phrase = self.phrase if self.phrase != pad_str else ""
            token_type = self.token_type if self.token_type != pad_str else ""
            task_id = self.task_id if self.task_id != pad_str else ""
            return f"[bar:{bar:>12}, pos:{position:>12}, dur:{duration:>12}, pit:{pitch:>12}, tmp:{tempo:>12}, vel:{velocity:>12}, lyr:{lyrics:>12}, group:{group:>12}, phrase:{phrase:>12}, token_t:{token_type:>12}, task_id:{task_id:>12}]"

    def define_vocabularies(self) -> None:
        self.vocabularies["bar"] = list(range(self.max_bar))
        self.vocabularies["position"] = self.position_bins
        self.vocabularies["duration"] = self.duration_bins
        self.vocabularies["pitch"] = list(self.pitch_range)
        self.vocabularies["tempo"] = list(self.tempo_range)
        self.vocabularies["velocity"] = list(self.velocity_range)
        self.vocabularies["lyrics"] = self.lyrics_bins
        self.vocabularies["group"] = list(self.group_range)
        self.vocabularies["phrase"] = list(self.phrase_range)
        self.vocabularies["token_type"] = self.token_type_bins
        self.vocabularies["task_id"] = list(self.task_id_bins)
        
        # add None tokens to vocabularies
        for field_name in self.vocabularies.keys():
            token_str = "<NONE>"
            self.vocabularies[field_name].append(token_str)

    def define_special_tokens(self) -> None:
        super().define_special_tokens()
        self.seg_token_str = "<SEG>"            # <REST> 
        self.seg_token = self.Token(*[self.seg_token_str] * len(self.field_names))
        self.special_token_str.append(self.seg_token_str)

    def set_special_token_ids(self) -> None:
        super().set_special_token_ids()
        self.seg_token_ids = self.convert_token_to_id(self.seg_token)
        
    def tokenize(self, midi: MidiFile, task_id: int = 0) -> Tuple[List[Token], np.ndarray]:
        def clean_text(text):
            cleaned_text = re.sub('[^a-zA-Z]', '', text)
            return cleaned_text.lower()

        def find_index(arr, target):
            for idx, ele in enumerate(arr):
                if target <= ele:
                    return idx
        
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        notes: List[Note] = midi.instruments[0].notes
        lyrics: List[Lyric] = midi.lyrics
        self.task_id = task_id

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
        
        lyric_indices_in_mixed_tokens = [index for index, token in enumerate(mixed_tokens) if type(token) == Lyric]
        mixed_token_indices = [(lyric_index, next_lyric_index) for lyric_index, next_lyric_index in zip(lyric_indices_in_mixed_tokens, lyric_indices_in_mixed_tokens[1:] + [len(mixed_tokens)])]

        note_duration_group = []
        for mixed_token_index in mixed_token_indices:
            start_index, end_index = mixed_token_index
            note_duration_group.append([mixed_tokens[note_index].end - mixed_tokens[note_index].start for note_index in range(start_index + 1, end_index)])

        note_group_dict = {}
        for group_index, mixed_token_index in enumerate(mixed_token_indices):
            start_index, end_index = mixed_token_index
            for i in range(start_index + 1, end_index):
                note_group_dict[notes.index(mixed_tokens[i])] = group_index

        segment_lyrics_indices, segment_note_indices = self.get_segment_indices(midi)

        # print note and lyrics mixed tokens
        # print([(index, value) for (index,value) in enumerate(mixed_tokens)])
        
        current_tempo_index = 0
        tempo_changes = self._get_tempo_changes(midi)
        
        tokens: List[self.Token] = []
        lyrics_tokens: List[self.Token] = []
        note_tokens: List[self.Token] = []
        
        mixed_tokens = lyrics + notes
        token_map = np.zeros(len(mixed_tokens), dtype=token_map_record_dtype)

        lyric_index = 0
        note_index = 0
        for token_index, token in enumerate(mixed_tokens):
            # Note token
            if type(token) == Note:
                if current_tempo_index < len(tempo_changes) - 1 and token.start >= tempo_changes[current_tempo_index + 1].time:
                    current_tempo_index += 1
                
                bar = (token.start // self.ticks_per_bar) % self.max_bar
                position = self._find_nearest(self.position_bins, token.start % self.ticks_per_bar)
                duration = self._find_nearest(self.duration_bins, token.end - token.start)
                pitch = token.pitch
                tempo = self._find_nearest(self.tempo_range, int(tempo_changes[current_tempo_index].tempo))
                velocity = self._find_nearest(self.velocity_range, token.velocity)
                lyrics = "<NONE>"
                group = note_group_dict[note_index]
                phrase = find_index(segment_note_indices, note_index)
                token_type = "note"
                task_id = self.task_id

                note_tokens.append(self.Token(bar, position, duration, pitch, tempo, velocity, lyrics, group, phrase, token_type, task_id))

                note_index += 1
            # Lyric token
            elif type(token) == Lyric:
                bar = "<NONE>"
                position = "<NONE>"
                duration = "<NONE>"
                pitch = "<NONE>"
                tempo = "<NONE>"
                velocity = "<NONE>"
                lyrics = clean_text(token.text)
                group = lyric_index
                phrase = find_index(segment_lyrics_indices, lyric_index)
                token_type = "lyrics"
                task_id = self.task_id

                lyrics_tokens.append(self.Token(bar, position, duration, pitch, tempo, velocity, lyrics, group, phrase, token_type, task_id))

                lyric_index += 1
            else:
                raise TypeError("Token type must be Note or Lyric.")

            if type(token) == Note:
                # +3 for the <BOS> token, <EOS> token and <BOS> token
                token_map[token_index] = (token_index + 3, token_index + 4)
            elif type(token) == Lyric:
                # +1 for the <BOS> token
                token_map[token_index] = (token_index + 1, token_index + 2)
                    
        tokens = [self.bos_token] + lyrics_tokens + [self.eos_token] + [self.bos_token] + note_tokens + [self.eos_token]

        return tokens, token_map

    def detokenize(self, tokens: List[Token]) -> MidiFile:
        midi = MidiFile()

        current_tempo = self.default_tempo
        midi.tempo_changes = [TempoChange(tempo=current_tempo, time=0)]
        
        note_group_dict = {}
        lyrics_group_dict = {}
        eos_flag = 0
        for token in tokens:
            if any([field == self.eos_token_str for field in token]):
                eos_flag += 1
                if eos_flag == 2:
                    break

            if any([field in self.special_token_str for field in token]):
                continue
            
            if token.token_type == "note":
                if type(token.bar) != int or type(token.duration) != int or type(token.position) != int or type(token.pitch) != int or type(token.velocity) != int or type(token.group) != int:
                    continue
                start = token.bar * self.ticks_per_bar + token.position
                end = start + token.duration
                pitch = token.pitch
                velocity = token.velocity
                note = Note(velocity=velocity, pitch=pitch, start=start, end=end)
                group = token.group

                if group not in note_group_dict.keys():
                    note_group_dict[group] = [note]
                else:
                    note_group_dict[group].append(note)

                # add tempo change if tempo changes
                # if token.tempo != current_tempo:
                #     current_tempo = token.tempo
                #     midi.tempo_changes.append(TempoChange(tempo=current_tempo, time=start))
            elif token.token_type == "lyrics":
                if type(token.group) != int:
                    continue
                text = token.lyrics
                group = token.group
                
                if group not in lyrics_group_dict.keys():
                    lyrics_group_dict[group] = [text]
                else:
                    lyrics_group_dict[group].append(text)

        midi_notes = []
        midi_lyrics = []

        for g in sorted(note_group_dict.keys()):
            if g not in lyrics_group_dict.keys():
                continue
            notes = sorted(note_group_dict[g], key=attrgetter('start'))
            start = notes[0].start
            midi_notes.extend(notes)
            midi_lyrics.append(Lyric(text=lyrics_group_dict[g][0], time=start))

        instrument = Instrument(program=0)
        instrument.notes.extend(midi_notes)
        midi.instruments.append(instrument)
        midi.lyrics.extend(midi_lyrics)

        return midi

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        return max(token.bar for token in tokens if isinstance(token.bar, int))
