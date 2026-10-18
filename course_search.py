from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Ten summaries of things covered in the course so far
sentences = [
    "FastAPI automatically validates request data using Pydantic models",
    "Pydantic schemas define what valid data looks like",
    "Error handling in FastAPI returns clear messages for invalid requests",
    "REST APIs expose endpoints that clients call over HTTP",
    "HTTP status codes communicate the outcome of a request",
    "Virtual environments isolate a project's dependencies from the system Python",
    "pip install adds packages into the active virtual environment",
    "SQLAlchemy maps Python classes to rows in a relational database",
    "JWT tokens carry signed claims for stateless authentication",
    "Sentence embeddings turn text into vectors so meaning can be compared numerically",
]

doc_embeddings = model.encode(sentences)

print("Course content search. Type a question, or 'quit' to exit.\n")

while True:
    query = input("Search (or 'quit'): ").strip()

    if query.lower() == "quit":
        break
    if not query:
        continue

    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)

    print("\nTop 3 results:")
    for rank, (idx, score) in enumerate(ranked[:3], 1):
        print(f"  {rank}. [{float(score):.4f}] {sentences[idx]}")
    print()