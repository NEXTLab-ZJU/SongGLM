# -----------------------------------------------------------------------------------------------------------------------------
# 1. extract ngrams in pre-trianing dataset with pitch peak & vowel and rhythm & vowel correlations.
# -----------------------------------------------------------------------------------------------------------------------------
#  1.1 pitch peak & vowel
python lexicon.py extract --length 12 \
                            --ngram_kind pitch_peak_vowel \
                            --midi_dir dataset/pre_processed/lyrics2melody_dataset  \
                            --ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_pitch_peak_vowel

#  1.2 rhythm & vowel
python lexicon.py extract --length 12 \
                            --ngram_kind rhythm_vowel \
                            --midi_dir dataset/pre_processed/lyrics2melody_dataset  \
                            --ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_rhythm_vowel

# -----------------------------------------------------------------------------------------------------------------------------
# 2. buid lexicon with pitch peak & vowel and rhythm & vowel correlations.
# -----------------------------------------------------------------------------------------------------------------------------
# 2.1 pitch peak & vowel
python lexicon.py build --length 12 \
                        --ngram_kind pitch_peak_vowel \
                        --ngram_dir  dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_pitch_peak_vowel \
                        --lexicon_path dataset/lyrics2melody_dataset/ngrams/lexicon/count/lexicon_pitch_peak_vowel.pkl

# 2.2 rhythm & vowel
python lexicon.py build --length 12 \
                        --ngram_kind rhythm_vowel \
                        --ngram_dir  dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_rhythm_vowel \
                        --lexicon_path dataset/lyrics2melody_dataset/ngrams/lexicon/count/lexicon_rhythm_vowel.pkl

# -----------------------------------------------------------------------------------------------------------------------------
# 3. rank ngrams with t-statistic score with pitch peak & vowel and rhythm & vowel correlations.
# -----------------------------------------------------------------------------------------------------------------------------
# 3.1 lyrics
python lexicon.py prepare --length 12 \
                          --ngram_kind pitch_peak_vowel \
                          --ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_pitch_peak_vowel \
                          --lexicon_path dataset/lyrics2melody_dataset/ngrams/lexicon/count/lexicon_pitch_peak_vowel.pkl \
                          --label_dir dataset/lyrics2melody_dataset/ngrams/lexicon/rank/label_pitch_peak_vowel

# 3.2 weighted relative_pitch
python lexicon.py prepare --length 12 \
                          --ngram_kind rhythm_vowel \
                          --ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/extract/ngram_rhythm_vowel \
                          --lexicon_path dataset/lyrics2melody_dataset/ngrams/lexicon/count/lexicon_rhythm_vowel.pkl \
                          --label_dir dataset/lyrics2melody_dataset/ngrams/lexicon/rank/label_rhythm_vowel
