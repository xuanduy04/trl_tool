import os
from pathlib import Path

import torch
from datasets import load_dataset

from trl import (
    GRPOToolConfig,
    GRPOToolTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    # get_peft_config,
    get_quantization_config,
)
from trl.generation_manager import LMGenerationConfig, LMGenerationManager
from trl.rewards import accuracy_reward, think_format_reward

from tools.calculator import calculator


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
from transformers import logging as hf_logging
hf_logging.set_verbosity_info()


SCRIPTS_DIR = Path(__file__).parent

def get_abs_path_from_scripts_dir(relpath: str) -> str:
    return str((SCRIPTS_DIR / relpath).resolve())


def main():
    parser = TrlParser((ScriptArguments, GRPOToolConfig, ModelConfig, LMGenerationConfig))
    script_args, training_args, model_args, generation_args = parser.parse_args_and_config()
    model_args.model_name_or_path = get_abs_path_from_scripts_dir(model_args.model_name_or_path)
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    DATASET_DIR = get_abs_path_from_scripts_dir("../../../../data/gsm8k_only_answer")
    # gsm8k_only_answer has columns: "text" and "label"
    train_dataset, eval_dataset = load_dataset(DATASET_DIR, split=["train[:1%]", "test[:1%]"])

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process and then provides the user with the answer. "
        "If the assistant needs to perform algebraic calculations, they can call the calculator"
        "with the <calculator></calculator> tags."
        "The reasoning process is enclosed within <think></think> tags."
        "The answer is enclosed within <answer></answer> tags."
        "Example: <think>\nThis is my reasoning.\n</think>\n<calculator>\n18+2\n</calculator>\n<answer>20</answer>."
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["text"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)
    
    _columns_to_remove = ["text"]
    train_dataset = train_dataset.remove_columns(_columns_to_remove).rename_column("label", "solution")
    eval_dataset = eval_dataset.remove_columns(_columns_to_remove).rename_column("label", "solution")

    ################
    # Training
    ################
    trainer = GRPOToolTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=get_peft_config(model_args),
    )

    generation_manager = LMGenerationManager(
        args=generation_args,
        processing_class=trainer.processing_class,
        tool=calculator,
        tool_first=False,
    )
    trainer.set_generation_manager(generation_manager)

    ckpt = os.environ.get("RESUME_CKPT", None)
    if ckpt is not None:
        ckpt = get_abs_path_from_scripts_dir(ckpt)
    trainer.train(resume_from_checkpoint=ckpt)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
