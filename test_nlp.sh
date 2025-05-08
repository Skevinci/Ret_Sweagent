#!/bin/bash

set -x
export N_GPUS=8
export BASE_MODEL="/nlp/data/sikaili/Qwen2.5-Coder-7B-Instruct"
export DATA_DIR="/nlp/data/sikaili/Ret_Sweagent/data/auto_sweagent"
export ROLLOUT_DIR="/nlp/data/sikaili/Ret_Sweagent/rollout"
export EXPERIMENT_NAME=test
export REWARD_PATH="/nlp/data/sikaili/Ret_Sweagent/verl/utils/reward_score/__init__.py"
export TMPDIR="/nlp/data/sikaili/tmp_ray"

/shared/sikaili/.conda/envs/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=8000 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=2 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/nlp/data/sikaili/Ret_Sweagent/checkpoints/test/test/global_step_1
    trainer.total_epochs=15 $@