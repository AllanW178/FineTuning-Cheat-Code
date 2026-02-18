### [RUN THE CODE BELOW FOR THE BASIC INSTALLATIONS; IN OTHER WORDS, IT'S JUST THE MODULES]

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft accelerate bitsandbytes





### [FULL FINE-TUNING PROCESS BELOW]:


from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


max_seq_length = 2048 # Mochi has short chats, so this is plenty
dtype = None          # Auto-detects your GPU settings
load_in_4bit = True   # CRITICAL: Fits the model into your 6GB VRAM


print("--- Loading Model... ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # The "Rank" (Intelligence of the adapter). 16 is standard.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("--- Processing Data... ---")
dataset = load_dataset("json", data_files="mochi_data.json", split="train")

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


print("--- Starting Training Engine... ---")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Keep low for 6GB VRAM
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 60 steps is enough for a small dataset
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

print("--- TRAINING COMPLETE! ---")
print("Saving your new Mochi model to 'lora_model' folder...")


model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")




