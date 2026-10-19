from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy.linalg import norm

model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences grouped by topic (we'll see if similarity reflects this)
sentences = [
    # Group 1: Python/API development
    "How to build a REST API with FastAPI",             # 0
    "Creating web services in Python",                   # 1
    "Python backend development with async support",     # 2
    # Group 2: Cooking
    "The best homemade pasta recipe with fresh tomatoes", # 3
    "How to make pizza dough from scratch",               # 4
    # Group 3: Space
    "NASA's latest Mars rover discovered water ice",     # 5
    "SpaceX launched a new satellite into orbit",        # 6
]

embeddings = model.encode(sentences)

# --- Build a similarity matrix ---
print("=== Cosine Similarity Matrix ===")
print("(Higher = more similar)\n")

# Short labels for readability
labels = ["FastAPI", "WebSvc", "PyBack", "Pasta", "Pizza", "Mars", "SpaceX"]

# Print header
print(f"{'':>8}", end="")
for label in labels:
    print(f"{label:>8}", end="")
print()

# Compute and print pairwise similarities
for i in range(len(sentences)):
    print(f"{labels[i]:>8}", end="")
    for j in range(len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[j]).item()
        print(f"{sim:>8.3f}", end="")
    print()

# --- Step 2: Finding the best match ---
print("\n=== Finding Best Matches ===")

queries = [
    "How do I create an API endpoint?",
    "What's a good recipe for Italian food?",
    "Tell me about recent space exploration",
]

for query in queries:
    query_emb = model.encode(query)
    scores = util.cos_sim(query_emb, embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    print(f"\nQuery: '{query}'")
    print(f"  Best match: [{best_score:.4f}] {sentences[best_idx]}")

# --- Step 3: Cosine vs. Euclidean ---
print("\n=== Cosine vs. Euclidean ===")

pairs = [
    (0, 1, "FastAPI vs WebServices (similar)"),
    (0, 3, "FastAPI vs Pasta (different)"),
    (3, 4, "Pasta vs Pizza (similar)"),
]

for i, j, label in pairs:
    cos_sim = util.cos_sim(embeddings[i], embeddings[j]).item()
    euc_dist = norm(embeddings[i] - embeddings[j])
    print(f"\n{label}:")
    print(f"  Cosine similarity: {cos_sim:.4f}  (higher = more similar)")
    print(f"  Euclidean distance: {euc_dist:.4f}  (lower = more similar)")