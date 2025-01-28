# Step 1: Clone and set up
git clone https://github.com/Sandeep-4469/LLM_PDF.git
cd LLM_PDF
python3 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Step 3: Prepare PDF
cp /path/to/your/sample.pdf data/raw/

# Step 4: Generate Q&A dataset
python src/dataset_generator.py --pdf_path data/raw/sample.pdf

# Step 5: Fine-tune the model
python src/finetune.py

# Step 6: Run the chatbot
python src/chat_interface.py