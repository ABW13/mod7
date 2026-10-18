
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "How do I build a REST API in Python?",
    "Creating a web service using FastAPI",
    "Best pizza recipe with mozzarella",
    "What is the capital of France?",
    "Building backend services with Python frameworks",
]

embeddings = model.encode(sentences)
query = embeddings[0]

print("Similarity to 'How do I build a REST API in Python?':\n")
for i, sentence in enumerate(sentences):
    similarity = util.cos_sim(query, embeddings[i]).item()
    print(f"  {similarity:.4f} — {sentence}")