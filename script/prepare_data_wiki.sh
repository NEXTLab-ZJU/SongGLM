# N = 4, Top 30% (default)
python prepare_data.py --kind octuple_like --granularity 64 --max_bar 128 --pitch_range 0 128 --velocity_range 0 128 --tempo_range 30 201 \
                       --total_task_num 1 --task_id 0 \
                       --ngram_length 4 --ngram_top_p 0.3 \
                       --midi_dir dataset/pre_processed/wiki \
                       --metadata_dir dataset/pre_processed/metadata.json \
                       --dataset_dir dataset/wiki/ngrams/pretrain/N4/train \
                       --config_dir dataset/wiki/ngrams/pretrain/N4
