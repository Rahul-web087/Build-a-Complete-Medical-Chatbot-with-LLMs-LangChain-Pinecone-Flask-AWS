from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec   # ✅ Correct Pinecone SDK import
from transformers import pipeline               # ✅ HuggingFace pipeline
from src.helper import download_hugging_face_embeddings  # your helper

# ----------------- Flask setup -----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ----------------- Load environment -----------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# ----------------- Pinecone setup -----------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# If index doesn’t exist, create it once (uncomment if needed)
# pc.create_index(
#     name="medical-chatbot",
#     dimension=768,   # depends on your embedding size
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )

index = pc.Index("medical-chatbot")

# ----------------- Embeddings setup -----------------
embeddings = download_hugging_face_embeddings()

# ----------------- HuggingFace model setup -----------------
#  Flan-T5 is instruction-tuned, better for Q&A
hf_model = pipeline("text2text-generation", model="google/flan-t5-large")

# ----------------- Routes -----------------

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def query_endpoint():
    body = request.get_json() or {}
    q = body.get("question") or body.get("q") or ""
    if not q:
        return jsonify({"error": "Send JSON {'question': '...'}"}), 400

    user_input = q.lower().strip()

    #   shortcuts conversation
    casual_responses = {
        "hi": "Hello! How can I help you today?",
        "hy": "How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hey there! How can I assist you?",
        "thanks": "You're welcome! Happy to help.",
        "thank you": "You're welcome! Glad I could assist.",
        "bye": "Goodbye! Take care and stay healthy.",
        "good morning": "Good morning! Wishing you a healthy and productive day.",
        "good night": "Good night! Rest well and take care."
    }

    if user_input in casual_responses:
        return jsonify({"answer": casual_responses[user_input], "retrieved": []})

    #  Fallback for slang greetings
    if any(word in user_input for word in ["yo", "sup", "what's up", "wassup", "bro"]):
        return jsonify({"answer": "Hey! I’m here — how can I help you today?", "retrieved": []})

    # General fallback for non-medical casual inputs
    if len(user_input.split()) <= 2 and not any(char.isalpha() for char in user_input):
        return jsonify({"answer": "I didn’t quite get that — could you rephrase?", "retrieved": []})

    # Normal flow for medical questions
    docs = get_top_k_docs(q, k=5)
    prompt = build_prompt(q, docs)

    response = hf_model(prompt, max_length=200)
    answer = response[0]["generated_text"].strip()

    return jsonify({"answer": answer, "retrieved": docs})

# ----------------- Helper functions -----------------

SYSTEM_PROMPT = (
    "You are a medical assistant. Use ONLY the provided context to answer. "
    "Be concise (max 3 sentences). If the answer is not in the context say 'I don't know based on the documents provided.'"
)

def get_top_k_docs(query, k=5):
    vector = embeddings.embed_query(query)
    resp = index.query(vector=vector, top_k=k, include_metadata=True)
    matches = resp.matches if hasattr(resp, "matches") else resp["matches"]
    docs = []
    for m in matches:
        meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {})
        text = meta.get("text") or meta.get("page_content") or meta.get("source_text") or meta.get("source") or ""
        docs.append({
            "score": getattr(m, "score", m.get("score") if isinstance(m, dict) else None),
            "metadata": meta,
            "text": text
        })
    return docs

def build_prompt(query, docs):
    context = "\n\n".join(f"- {d['text']}" for d in docs if d['text'])
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

# ----------------- Run app -----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
