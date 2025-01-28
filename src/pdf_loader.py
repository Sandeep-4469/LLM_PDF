import yaml
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_and_split_pdf(pdf_path):
    config = load_config()
    
    # Extract text
    reader = PdfReader(pdf_path)
    text = " ".join([page.extract_text() for page in reader.pages])
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["processing"]["chunk_size"],
        chunk_overlap=config["processing"]["chunk_overlap"]
    )
    return text_splitter.split_text(text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, required=True)
    args = parser.parse_args()
    
    chunks = load_and_split_pdf(args.pdf_path)
    print(f"Loaded {len(chunks)} chunks from PDF")