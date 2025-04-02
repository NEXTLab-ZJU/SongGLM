import argparse
import os
from glob import glob

from pretrain.ngram import NgramExtractor, NotePitchPeakNgramList, NoteRhythmNgramList, LyricVowelNgramList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", type=str, choices=["metadata", "extract", "build", "prepare"])
    parser.add_argument("--midi_dir", type=str)
    parser.add_argument("--ngram_dir", type=str)
    parser.add_argument("--metadata_dir", type=str)
    parser.add_argument("--lexicon_path", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--ngram_kind", type=str, choices=["pitch_peak_vowel", "rhythm_vowel"])
    args = parser.parse_args()

    if args.ngram_kind == "pitch_peak_vowel":
        note_ngram_type = NotePitchPeakNgramList
        lyric_ngram_type = LyricVowelNgramList
    elif args.ngram_kind == "rhythm_vowel":
        note_ngram_type = NoteRhythmNgramList
        lyric_ngram_type = LyricVowelNgramList

    if args.subcommand == "metadata":
        assert args.midi_dir is not None and args.metadata_dir is not None
        midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
        print(f"Find {len(midi_files)} midi files")

        extractor = NgramExtractor(n_range=(1, args.length), note_ngram_type=note_ngram_type, lyric_ngram_type=lyric_ngram_type)
        extractor.get_metadata(midi_files, args.metadata_dir)
    elif args.subcommand == "extract":
        assert args.midi_dir is not None and args.ngram_dir is not None
        midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
        print(f"Find {len(midi_files)} midi files")
        os.makedirs(args.ngram_dir, exist_ok=True)

        extractor = NgramExtractor(n_range=(1, args.length), note_ngram_type=note_ngram_type, lyric_ngram_type=lyric_ngram_type)
        extractor.extract_ngrams(midi_files, args.ngram_dir)
    elif args.subcommand == "build":
        assert args.ngram_dir is not None and args.lexicon_path is not None
        ngram_files = glob(os.path.join(args.ngram_dir, "*.pkl"))
    
        extractor = NgramExtractor(n_range=(1, args.length), note_ngram_type=note_ngram_type, lyric_ngram_type=lyric_ngram_type)
        extractor.build_lexicon(ngram_files, args.lexicon_path, ngram_type=args.ngram_kind)
    elif args.subcommand == "prepare":
        assert args.ngram_dir is not None and args.label_dir is not None and args.lexicon_path is not None
        ngram_files = glob(os.path.join(args.ngram_dir, "*.pkl"))
        os.makedirs(args.label_dir, exist_ok=True)
        
        extractor = NgramExtractor(n_range=(1, args.length), note_ngram_type=note_ngram_type, lyric_ngram_type=lyric_ngram_type)
        extractor.prepare_ngram_labels(ngram_files, args.label_dir, args.lexicon_path)
