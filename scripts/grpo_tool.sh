BASE_MODEL='Qwen2.5-0.5B'

# REMEMBER TO FIX 'accelerate_configs' file when changing CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="4,7" \
PYTHONPATH=. \
accelerate launch \
    --config_file ./accelerate_configs/fsdp2.yaml \
    ./main/main_grpo_tool.py \
    --model_name_or_path ../../../../../NLP_CORE/BaseModels/${BASE_MODEL} \
    --output_dir ${BASE_MODEL}-grpo-tool \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 4096 \
    --max_completion_length 800 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8 \
    --num_generations 8 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --beta 0.0 \
    --importance_sampling_level token \
    --loss_type grpo \
    --log_completions \
    --num_completions_to_print 0 \
    --report_to none
