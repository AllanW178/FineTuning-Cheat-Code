# (STEP 1)
%%capture
import torch
major_version, minor_version = torch.cuda.get_device_capability()
# Must install Unsloth specifically for Colab
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# Support for the newest models
if major_version >= 8:
    !pip install --no-deps packaging
    !pip install flash-attn --no-build-isolation




# ------------------------------------------------------------------------------------
# (STEP 2)
# REMEMBER TO DRAG YOUR {dynamic_character_dataset}.json into the Google Colab folder.

# ------------------------------------------------------------------------------------


# (STEP 3 - TRAIN!)


from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CONFIGURATION ---
max_seq_length = 2048
dtype = None # Auto-detects (Float16 for T4, Bfloat16 for Ampere)
load_in_4bit = True # Essential for Colab's 16GB VRAM (T4)

# 1. LOAD MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. ADD LORA ADAPTERS
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    random_state = 3407,
)

# 3. PREPARE DATA
# Ensure 'mochi_data.json' is uploaded to the root folder on the left
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

# 4. TRAIN
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
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

# 5. SAVE & DOWNLOAD
# In Colab, if you don't download, you lose the file when you close the tab.
print("Saving model locally in Colab...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Option A: Save to GGUF (for Ollama usage later)
# model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")

print("DONE! You can now zip and download the 'lora_model' folder from the files pane.")




# ------------------------------------------------------------------------------------


# (Google Colab's FINAL STEP: Step 4 - ZIP IT!)


!zip -r mochi_lora.zip lora_model
from google.colab import files
files.download('mochi_lora.zip')


# ------------------------------------------------------------------------------------


# FINAL STEP TO FINISH THE MODEL INSTALLATION (Step 5):



from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# 1. SETUP
# We use the exact same settings as training to avoid confusion
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("--- 1. Loading the Main Brain (Llama-3) ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # The base model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("--- 2. Applying Mochi's Personality ---")
# This loads your specific folder on top of the base model
model.load_adapter("mochi_lora/lora_model")

# Set the model to "Inference Mode" (Makes it faster, turns off training features)
FastLanguageModel.for_inference(model)

print("--- 3. Mochi is Awake! (Type 'quit' to exit) ---")

# We use the chat template so Mochi knows who is speaking
def chat_with_mochi(user_input):
    messages = [
        {"role": "user", "content": user_input},
    ]
    
    # Prepare the input for the model
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Tells Mochi "It's your turn now"
        return_tensors = "pt",
    ).to("cuda")

    # Streamer makes the text appear word-by-word (like ChatGPT)
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)

    # Generate the response
    _ = model.generate(
        input_ids = inputs,
        streamer = text_streamer,
        max_new_tokens = 128, # Cap the response length
        use_cache = True,
        temperature = 1.5, # Creativity: 1.5 is high (good for Mochi), 0.1 is robotic
        min_p = 0.1,
    )

while True:
    user_input = input("\nYOU: ")
    if user_input.lower() == "quit":
        break
    print("MOCHI: ", end="")
    chat_with_mochi(user_input)
    print("") # New line


# ------------------------------------------------------------------------------------



    
