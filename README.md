# Fine-tuning T5 for Grade School Math (GSM8K)

This project demonstrates fine-tuning a T5 model (`t5-small`) on the GSM8K dataset to solve grade-school-level math word problems with detailed multi-step reasoning.

## 📚 Dataset
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k): Contains math questions and step-by-step solutions.

## Features
- Text-to-text model (T5) trained on `question -> step-by-step answer`
- Gradio interface for real-time inference
- Tokenization and formatting of GSM8K using HuggingFace `datasets`

## Tech Stack
- PyTorch, HuggingFace Transformers, Gradio, GSM8K
- Optional fine-tuning with Google Colab

## 📷 Demo Screenshot
![Screenshot 2025-06-24 012936](https://github.com/user-attachments/assets/3618b259-eb92-4a2f-8d33-571d2b4dd13f)


## Try it
Clone the repo and run:

```bash
pip install -r requirements.txt
python train.py
python app_gradio.py
