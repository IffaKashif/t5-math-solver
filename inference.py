from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

#Load T5 Model and Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained("./t5-math-model")

#Inference Function
def solve_math_problem(problem_text):
    inputs = tokenizer("Solve: " + problem_text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

