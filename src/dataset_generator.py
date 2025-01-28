import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

def generate_qa_pairs(text_chunks):
    config = yaml.safe_load(open("config/config.yaml"))
    
    # Load model for Q&A generation
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        device_map=config["model"]["device"],
        torch_dtype=torch.float16
    )
    
    qa_data = []
    for chunk in text_chunks:
        prompt = f"""
        Generate 2 questions and answers based on the following text. Use this format:
        Q: [question]
        A: [answer]

        Text: {chunk}
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse Q&A from response
        qa_list = response.split("Text:")[-1].strip().split("\n")
        for qa in qa_list:
            if "Q:" in qa and "A:" in qa:
                q, a = qa.split("A:")
                qa_data.append({"question": q.replace("Q:", "").strip(), "answer": a.strip()})
    
    return pd.DataFrame(qa_data)

if __name__ == "__main__":
    from pdf_loader import load_and_split_pdf
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, required=True)
    args = parser.parse_args()
    
    chunks = load_and_split_pdf(args.pdf_path)
    qa_df = generate_qa_pairs(chunks[:5])  # Use first 5 chunks for demo
    qa_df.to_csv("data/processed/qa_dataset.csv", index=False)
    print(f"Generated {len(qa_df)} Q&A pairs")