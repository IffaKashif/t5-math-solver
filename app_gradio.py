import gradio as gr
from inference import solve_math_problem


#Gradio Inference GUI
def gradio_interface(question):
    return solve_math_problem(question)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=4, label="Enter a Grade School Math Problem"),
    outputs=gr.Textbox(label="Model's Answer"),
    title="T5 Math Problem Solver (Demo)",
    description="Fine-tuned on GSM8K subset with multi-step reasoning."
)

demo.launch()
