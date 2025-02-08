import torch
from transformers import MarianMTModel, MarianTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import datasets
from transformers import DataCollatorForSeq2Seq

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Load model and tokenizer
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    
    # Add LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.to(device)

    # Load dataset
    dataset = datasets.load_dataset("csv", data_files={
        "train": "data/samanantar/en-hi/train.csv",
        "validation": "data/samanantar/en-hi/valid.csv"
    })

    # Tokenization
    def preprocess_function(examples):
        inputs = [ex for ex in examples["en"]]
        targets = [ex for ex in examples["hi"]]
        model_inputs = tokenizer(
            inputs, max_length=128, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=128, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/fine-tuned-marianmt-en-hi",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        save_strategy="epoch",
        logging_steps=100,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()
    model.save_pretrained("models/fine-tuned-marianmt-en-hi")

if __name__ == "__main__":
    train_model()