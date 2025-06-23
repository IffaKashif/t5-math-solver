from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")

# Use a small subset (e.g., 100 training, 20 testing)
train_data = dataset["train"]
#.select(range(100))
test_data = dataset["test"]
#.select(range(20))

# Format input/output
def format_example(example):
    return {
        "input_text": "Solve: " + example["question"],
        "target_text": example["answer"]
    }

train_data = train_data.map(format_example)
test_data = test_data.map(format_example)
