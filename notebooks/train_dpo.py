import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  # Specify the GPUs to use


import torch
import torch.distributed as dist
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import get_peft_model, LoraConfig

# def setup_distributed():
#     """Initialize the distributed process group."""
#     if not dist.is_initialized():
#         dist.init_process_group(backend="nccl")  # NCCL for GPUs
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#         torch.cuda.set_device(rank)
#         print(f"Rank {rank}/{world_size} initialized")
#     else:
#         print("Process group already initialized")

# setup_distributed()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")



lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    logging_steps=10,
    use_logits_to_keep=True,
    fp16=True,
    per_device_train_batch_size=1,
    max_prompt_length=100,
    max_completion_length=50,
    ddp_backend="nccl",
    fsdp="full_shard",
    # precompute_ref_log_probs=True,
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    peft_config=lora_config,
)
trainer.train()
