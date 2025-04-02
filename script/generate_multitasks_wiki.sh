# N12
python main.py predict --config config/model/base.yaml \
                       --config config/predict/generate_clm_wiki.yaml \
                       --trainer.default_root_dir experiment/mixed/model/multitasks_4_cr50_n4/pretrain \
                       --ckpt_path experiment/mixed/model/multitasks_4_cr50_n4/finetune_clm_wiki/lightning_logs/version_0/checkpoints/best_val.ckpt \
                       --trainer.callbacks.output_dir experiment/mixed/model/multitasks_4_cr50_n4/generate_clm_wiki