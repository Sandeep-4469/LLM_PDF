from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

def generate_answer(question, tokenizer, model):
    prompt = f"### Instruction: {question}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./fine_tuned_model")
    args = parser.parse_args()
    
    tokenizer, model = load_model(args.model_path)
    answer = generate_answer(args.question, tokenizer, model)
    print(f"Question: {args.question}")
    print(f"Answer: {answer.split('Response:')[-1].strip()}")