CUDA_VISIBLE_DEVICES="4,7" \
PYTHONPATH=. \
accelerate launch \
    --config_file ./accelerate_configs/fsdp2.yaml \
    ./main/main_grpo.py \
    --model_name_or_path ../../../../../NLP_CORE/BaseModels/Qwen2.5-0.5B \
    --output_dir Qwen2.5-0.5B-grpo \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 800 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --importance_sampling_level token \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --loss_type grpo \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8 \
    --report_to none
