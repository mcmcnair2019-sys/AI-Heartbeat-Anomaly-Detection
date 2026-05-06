import os, glob, re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline
import torch
 
app = Flask(__name__)
CORS(app)
 
#  Load LLM 
print("Loading LLM...")
device = 0 if torch.cuda.is_available() else -1
llm = hf_pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device=device,
    max_new_tokens=300,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print(f"LLM loaded on {'GPU' if device == 0 else 'CPU'}")  # Allow the website to call this API from a different port
 
#  Paths 
DOCS_DIR  = r"C:\Users\matty\ksu\AI Development\AI Capstone\Docs"
DEMOS_DIR = r"C:\Users\matty\ksu\AI Development\AI Capstone\Demos"
 
TOP_K      = 5
TOP_RERANK = 2
 
#  Load models and knowledgebase on startup 
print("Loading embedding model...")
embed_model = SentenceTransformer("thenlper/gte-large")
 
print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
print("Loading knowledgebase...")
all_texts = []
for path in sorted(glob.glob(os.path.join(DOCS_DIR,  "*.txt"))) + \
            sorted(glob.glob(os.path.join(DEMOS_DIR, "*.txt"))):
    with open(path) as f:
        all_texts.append(f.read())
 
def chunk(text):
    if "TASK:" in text:
        return [text.strip()]
    parts = text.split(". ")
    return [" ".join(parts[i:i+2]) for i in range(0, len(parts), 2)]
 
chunks = []
for t in all_texts:
    chunks.extend(chunk(t))
 
print(f"Encoding {len(chunks)} chunks...")
embeddings = embed_model.encode(chunks, show_progress_bar=False)
tokenized  = [c.lower().split() for c in chunks]
bm25       = BM25Okapi(tokenized)
print("API ready.")
 
#  Helpers ─
def extract_features(signal):
    x = np.array(signal)
    return {
        "mean":               float(np.mean(x)),
        "variance":           float(np.var(x)),
        "fft_mean_magnitude": float(np.mean(np.abs(np.fft.rfft(x)))),
        "signal_text":        ", ".join(f"{v:.2f}" for v in x),
    }
 
def hybrid_retrieve(query):
    q_emb       = embed_model.encode([query])
    cos_scores  = cosine_similarity(q_emb, embeddings)[0]
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    combined = 0.5 * cos_scores + 0.5 * bm25_scores
    return np.argsort(combined)[::-1][:TOP_K].tolist()
 
def rerank_chunks(query, indices):
    pairs  = [[query, chunks[i]] for i in indices]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, indices), reverse=True)
    return [chunks[idx] for _, idx in ranked[:TOP_RERANK]]
 
def parse_label(text):
    match = re.search(r'LABEL:\s*(Normal|PVC)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    if re.search(r'\bPVC\b', text):
        return "PVC"
    if re.search(r'\bNormal\b', text, re.IGNORECASE):
        return "Normal"
    return "Unknown"
 
def parse_confidence(text):
    match = re.search(r'CONFIDENCE:\s*(\d+)', text)
    return int(match.group(1)) if match else -1
 
#  Route 
@app.route("/classify", methods=["POST"])
def classify():
    try:
        data   = request.get_json()
        signal = data.get("signal", [])
 
        if len(signal) < 10:
            return jsonify({"error": "Signal too short. Need at least 10 samples."}), 400
 
        # Extract features
        feats = extract_features(signal)
 
        # Retrieve context
        query      = feats["signal_text"][:200]
        candidates = hybrid_retrieve(query)
        top        = rerank_chunks(query, candidates)
        context    = "\n\n---\n\n".join(top)
 
        # Build prompt
        prompt = f"""You are a cardiologist analyzing ECG data.
TASK: Classify the heartbeat as Normal or PVC.
 
ECG SIGNAL: {feats['signal_text'][:300]}
 
FEATURES:
Mean: {feats['mean']:.4f}
Variance: {feats['variance']:.4f}
FFT Mean Magnitude: {feats['fft_mean_magnitude']:.4f}
 
RETRIEVED KNOWLEDGE:
{context}
 
Analyze the ECG and provide:
1. Step-by-step reasoning
2. Final label (LABEL: Normal or LABEL: PVC)
3. Confidence score (CONFIDENCE: 0-100)
"""
 
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        result = llm(messages, do_sample=False, temperature=None, top_p=None)
        raw = result[0]['generated_text'][-1]['content'].strip()
 
        return jsonify({
            "label":      parse_label(raw),
            "confidence": parse_confidence(raw),
            "reasoning":  raw,
            "features":   {
                "mean":     round(feats["mean"], 4),
                "variance": round(feats["variance"], 4),
                "fft":      round(feats["fft_mean_magnitude"], 4),
            },
            "retrieved_chunks": top,
        })
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)