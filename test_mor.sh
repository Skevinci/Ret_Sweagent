#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export N_GPUS=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_MODEL="/shared/sikaili/Qwen2.5-0.5B"
export DATA_DIR="/home1/s/sikaili/data/auto_sweagent"
export ROLLOUT_DIR="/home1/s/sikaili/Ret_Sweagent/rollout"
export EXPERIMENT_NAME=test
export REWARD_PATH="/home1/s/sikaili/Ret_Sweagent/verl/utils/reward_score/__init__.py"

/shared/sikaili/.conda/envs/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=5000 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=21144 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=_default_compute_score \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.logger=['console','wandb'] \
    trainer.project_name='test' \
    trainer.experiment_name='test' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    trainer.test_freq=2 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=15 $@