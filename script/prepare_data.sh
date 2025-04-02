# N = 4, Top 30% (default)
python prepare_data.py --kind octuple_like --granularity 64 --max_bar 128 --pitch_range 0 128 --velocity_range 0 128 --tempo_range 30 201 \
                       --total_task_num 1 --task_id 0 \
                       --ngram_length 4 --ngram_top_p 0.3 \
                       --midi_dir dataset/pre_processed/lyrics2melody_dataset \
                       --metadata_dir dataset/pre_processed/metadata.json \
                       --pitch_peak_vowel_ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/rank/label_pitch_peak_vowel \
                       --rhythm_vowel_ngram_dir dataset/lyrics2melody_dataset/ngrams/lexicon/rank/label_rhythm_vowel \
                       --dataset_dir dataset/lyrics2melody_dataset/ngrams/pretrain/N4/train \
                       --config_dir dataset/lyrics2melody_dataset/ngrams/pretrain/N4
