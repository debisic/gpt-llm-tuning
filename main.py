"""
This module contains basic mathematical operations.
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM , AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


# The model that you want to train from the Hugging Face hub
MODEL_NAME = "openai-community/gpt2"

# The instruction dataset to use
DATASET_NAME = "tatsu-lab/alpaca"

# Fine-tuned model name
NEW_MODEL = "gpt2-fine-tuned"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
LORA_R = 64

# Alpha parameter for LoRA scaling
LORA_ALPHA = 16

# Dropout probability for LoRA layers
LORA_DROPOUT = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
USE_4BIT = True

# Compute dtype for 4-bit base models
BNB_4BIT_COMPUTE_DTYPE = "float16"

# Quantization type (fp4 or nf4)
BNB_4BIT_QUANT_TYPE = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
USE_NESTED_QUANT = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
OUTPUT_DIR = "./results"

# Number of training epochs
NUM_TRAIN_EPOCHS = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
FP16 = False
BF16 = False

# Batch size per GPU for training
PER_DEVICE_TRAIN_BATCH_SIZE = 4

# Batch size per GPU for evaluation
PER_DEVICE_EVAL_BATCH_SIZE = 4

# Number of update steps to accumulate the gradients for
GRADIENT_ACCUMULATION_STEPS = 1

# Enable gradient checkpointing
GRADIENT_CHECKPOINTING = True

# Maximum gradient normal (gradient clipping)
MAX_GRAD_NORM = 0.3

# Initial learning rate (AdamW optimizer)
LEARNING_RATE = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
WEIGHT_DECAY = 0.001

# Optimizer to use
OPTIM = "paged_adamw_32bit"

# Learning rate schedule
LR_SCHEDULER_TYPE = "cosine"

# Number of training steps (overrides num_train_epochs)
MAX_STEPS = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
WARMUP_RATIO = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
GROUP_BY_LENGTH = True

# Save checkpoint every X updates steps
SAVE_STEPS = 0

# Log every X updates steps
LOGGING_STEPS = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
MAX_SEQ_LENGTH = None

# Pack multiple short examples in the same input sequence to increase efficiency
PACKING = False

# Load the entire model on the GPU 0
DEVICE_MAP = {"": 0}




# Load dataset (you can process it here)
#dataset = load_from_disk(DATASET_NAME)
dataset = load_dataset("tatsu-lab/alpaca")
# dataset = ds.shuffle(seed = 42)

# Load tokenizer and model with QLoRA configuration
COMPUTE_DTYPE = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE ,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)

# Check GPU compatibility with bfloat16
if COMPUTE_DTYPE == torch.float16 and USE_4BIT:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE ,
    gradient_accumulation_steps= GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIM,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    bf16=BF16,
    max_grad_norm=MAX_GRAD_NORM,
    max_steps=MAX_STEPS,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=GROUP_BY_LENGTH ,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    #packing=packing,
)

# Train model
trainer.train()

trainer.save_model("./runs/gpt2-fine-tuned")
