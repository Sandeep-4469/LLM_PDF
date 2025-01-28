from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

def format_prompt(message, history):
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"[INST] {user_msg} [/INST] {bot_msg} </s>"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate_response(message, history, tokenizer, model):
    prompt = format_prompt(message, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

def create_chat_interface(model_path):
    tokenizer, model = load_model(model_path)
    
    chat_interface = gr.ChatInterface(
        fn=lambda message, history: generate_response(message, history, tokenizer, model),
        title="PDF QA Chatbot",
        examples=["What is the document about?"],
        css=".gradio-container {background-color: #f0f2f6}"
    )
    
    return chat_interface

if __name__ == "__main__":
    interface = create_chat_interface("./fine_tuned_model")
    interface.launch(share=True)  # Set share=False for local only