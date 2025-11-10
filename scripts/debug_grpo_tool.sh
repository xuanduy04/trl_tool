export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH=. \
accelerate launch \
    --config_file ./accelerate_configs/debug_cpu.yaml \
    ./main/main_grpo_tool.py \
    --model_name_or_path ../../../../models/Qwen2.5-0.5B-Instruct \
    --output_dir grpo_tool-Qwen2.5-0.5B-it \
    --learning_rate 1e-5 \
    --fp16 \
    --max_prompt_length 4096 \
    --max_completion_length 800 \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 2 \
    --importance_sampling_level token \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8