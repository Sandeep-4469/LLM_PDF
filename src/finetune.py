import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def finetune():
    config = load_config()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        device_map=config["model"]["device"],
        torch_dtype=torch.float16
    )
    
    # Load dataset
    qa_df = pd.read_csv("data/processed/qa_dataset.csv")
    dataset = Dataset.from_pandas(qa_df)
    
    # Format dataset for instruction tuning
    def format_instruction(sample):
        return {
            "text": f"### Instruction: {sample['question']}\n### Response: {sample['answer']}"
        }
    dataset = dataset.map(format_instruction)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ) if config["model"]["use_lora"] else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=config["training"]["learning_rate"],
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=config["training"]["max_seq_length"],
        peft_config=peft_config,
        dataset_text_field="text"
    )
    
    trainer.train()
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    finetune()