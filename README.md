# 🩺 Medical Chatbot with Flask, Pinecone & HuggingFace

A stylish and interactive **medical assistant chatbot** built using:
- **Flask** for the backend
- **Pinecone** for vector search (retrieval of medical documents)
- **HuggingFace Flan‑T5** for instruction‑tuned answers
- HTML/CSS

# Techstack Used:
Python
LangChain
Flask
GPT
Pinecone

## ✨ Features
- ✅ Retrieval‑augmented answers from medical context
- ✅ Instruction‑tuned model (Flan‑T5) for concise, accurate responses
- ✅ Casual  conversation shortcuts (hi, hello, thanks, bye, etc.)

- 
## 📂 Project Structure
project/ ├── app.py                # Flask backend ├── requirements.txt                  # Deployment start command ├── templates/ │   └── chat.html          # Frontend HTML ├── static/ │   ├── style.css          # CSS styling │



# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
Create a .env file
PINECONE_API_KEY=your_pinecone_api_key

# Run locally
python app.py
