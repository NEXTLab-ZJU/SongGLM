# multi-task cr15% + cr50% | N 4 | 
# 1. define masking strategy
# 2. create data collator for batch data
# 3. create data loader for training

# export NCCL_P2P_DISABLE=1

python main.py fit --config config/trainer/trainer.yaml \
                   --config config/model/base.yaml \
                   --config config/task/mixed/multitasks_4_cr50_n4.yaml \
                   --trainer.default_root_dir experiment/mixed/model/multitasks_4_cr50_n4/pretrain