import json
import joblib
from sentence_transformers import SentenceTransformer

# Load fallback Q&A
with open("fallback_qna.json", "r", encoding="utf-8") as f:
    fallback_data = json.load(f)

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en")

embeddings = []

for item in fallback_data:
    for question in item["questions"]:
        vec = model.encode([question])[0]
        embeddings.append((question, vec, item["answer"]))

# Save the embeddings as .pkl
joblib.dump(embeddings, "fallback_embeddings.pkl")

print("✅ Embeddings generated and saved to fallback_embeddings.pkl")
