# cr = 50%
python main.py fit --config config/trainer/trainer_finetune.yaml \
                   --config config/model/base.yaml \
                   --config config/finetune/clm_wiki.yaml \
                   --trainer.default_root_dir experiment/mixed/model/multitasks_4_cr50_n4/finetune_clm_wiki \
                   --load_from_checkpoint experiment/mixed/model/multitasks_4_cr50_n4/pretrain/lightning_logs/version_0/checkpoints/best.ckpt