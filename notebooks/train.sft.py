import os
import torch
from torch import distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def setup_distributed():
    """Initialize the distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # NCCL for GPUs
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        print(f"Rank {rank}/{world_size} initialized")
    else:
        print("Process group already initialized")


setup_distributed()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_length=512,
    output_dir="logs/test",
    # ddp_backend="nccl",
    # fsdp="full_shard",
)
training_args.set_training(batch_size=1)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config
)
trainer.train()

dist.destroy_process_group()