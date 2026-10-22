import chromadb
from chromadb.utils import embedding_functions

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./knowledge_base_db")

collection = client.get_or_create_collection(
    name="my_knowledge",
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    "SQLAlchemy is an ORM that maps Python classes to database tables.",
    "Database migrations apply schema changes to an existing database over time.",
    "A database session tracks pending changes and commits them as one transaction.",
    "FastAPI validates incoming request data automatically using Pydantic models.",
    "HTTP status codes communicate the outcome of a request to the client.",
    "JWT tokens provide stateless authentication for REST APIs.",
    "Error handling in FastAPI returns structured messages for invalid requests.",
    "Streamlit re-runs the entire script every time a user interacts with a widget.",
    "CSS Flexbox arranges child elements in a row or column with flexible sizing.",
    "The DOM is the browser's tree representation of an HTML page.",
    "Embeddings convert text into numerical vectors that capture semantic meaning.",
    "Cosine similarity measures the angle between two vectors.",
    "Chunking divides a long document into smaller pieces before embedding.",
    "A vector database stores embeddings and supports similarity search over them.",
    "Metadata filtering narrows a search to a subset of documents before ranking.",
    "A virtual environment isolates a project's dependencies from the system Python.",
    "Git commits record snapshots of a project's files.",
    "pytest discovers and runs test functions, reporting pass and fail results.",
]

metadatas = [
    {"module": "3", "topic": "database"},
    {"module": "3", "topic": "database"},
    {"module": "3", "topic": "database"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "security"},
    {"module": "5", "topic": "api"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "retrieval"},
    {"module": "4", "topic": "tooling"},
    {"module": "4", "topic": "tooling"},
    {"module": "4", "topic": "tooling"},
]

ids = [f"kb_{i:02d}" for i in range(len(documents))]


def search(query, module=None, n_results=5):
    where = {"module": module} if module else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "similarity": 1 - results["distances"][0][i],
            "module": results["metadatas"][0][i]["module"],
            "topic": results["metadatas"][0][i]["topic"],
            "document": results["documents"][0][i],
        })
    return hits


def show(query, module=None):
    label = f"module {module} only" if module else "no filter"
    print("\n" + "=" * 78)
    print(f"Query: {query!r}  ({label})")
    print("=" * 78)

    hits = search(query, module=module)

    if not hits:
        print("  (no results)")
        return

    for rank, hit in enumerate(hits, start=1):
        print(
            f"  {rank}. [dist {hit['distance']:.4f} | "
            f"sim {hit['similarity']:.4f}] "
            f"[{hit['id']}] (M{hit['module']}, {hit['topic']})"
        )
        print(f"     {hit['document']}")


collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

print(f"Collection '{collection.name}' ready.")
print(f"Documents in collection: {collection.count()}")

module_counts = {}
for meta in metadatas:
    module_counts[meta["module"]] = module_counts.get(meta["module"], 0) + 1
print("By module:", ", ".join(
    f"M{m}: {c}" for m, c in sorted(module_counts.items())
))

show("How do I keep my project's packages separate from the system install?")
show("How does the app react when a user clicks something?", module="6")
show("finding the closest meaning between two pieces of text")
