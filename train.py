from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

#Load T5 Model and Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#Tokenize Data
def tokenize(example):
    inputs = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_tokenized = train_data.map(tokenize, remove_columns=train_data.column_names)
test_tokenized = test_data.map(tokenize, remove_columns=test_data.column_names)


#Training Arguments & Trainer
training_args = TrainingArguments(
    output_dir="./t5-math-model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

trainer.train()


