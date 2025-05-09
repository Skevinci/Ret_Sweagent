#!/bin/bash

set -x
export N_GPUS=8
export BASE_MODEL="/home/ec2-user/Qwen2.5-Coder-7B-Instruct"
export DATA_DIR="/home/ec2-user/data/auto_sweagent"
export ROLLOUT_DIR="/home/ec2-user/Ret_Sweagent/rollout"
export EXPERIMENT_NAME=ec2-experiment
export REWARD_PATH="/home/ec2-user/Ret_Sweagent/verl/utils/reward_score/__init__.py"

/home/ec2-user/miniconda3/envs/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=5000 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=21144 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=112768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=112768 \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=_default_compute_score \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ec2_train' \
    trainer.experiment_name='train_code_0506' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=15 $@

    # trainer.resume_mode=resume_path \
    # trainer.resume_from_path="/home/ec2-user/Ret_Sweagent/checkpoints/ec2_train/train_0506/global_step_6" \