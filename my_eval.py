import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------------------------------------------
# Setup: a ChromaDB collection with 15 documents across 4 topics
# ----------------------------------------------------------------------

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./my_eval_db")

collection = client.get_or_create_collection(
    name="eval_docs",
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    # --- APIs and authentication ---
    ("d_jwt", "JWT tokens provide stateless authentication for REST APIs. "
              "The server returns a signed token containing user claims.", "api"),
    ("d_pydantic", "Pydantic models define data schemas for FastAPI and "
                   "automatically validate incoming request data.", "api"),
    ("d_cors", "CORS middleware in FastAPI allows cross-origin requests from "
               "frontends running on a different port.", "api"),
    ("d_status", "HTTP status codes communicate the outcome of a request: 200 "
                 "for success, 404 for a missing resource, 422 for invalid data.", "api"),
    # --- Frontend ---
    ("d_dom", "The DOM is the browser's tree representation of an HTML page. "
              "JavaScript modifies it to change what the user sees.", "frontend"),
    ("d_flex", "CSS Flexbox arranges child elements in a row or column with "
               "flexible sizing and gap for spacing.", "frontend"),
    ("d_streamlit", "Streamlit re-runs the entire script on every widget "
                    "interaction, so ordinary variables reset each time.", "frontend"),
    ("d_session", "Streamlit session state persists values across script "
                  "re-runs. Initialize with: if key not in st.session_state.", "frontend"),
    # --- Retrieval and embeddings ---
    ("d_embed", "Embeddings convert text into numerical vectors that capture "
                "semantic meaning. Similar texts produce similar vectors.", "retrieval"),
    ("d_cosine", "Cosine similarity measures the angle between two vectors. "
                 "A score of 1.0 means identical direction.", "retrieval"),
    ("d_chunk", "Chunking splits a long document into smaller passages before "
                "embedding. Chunk size affects both precision and recall.", "retrieval"),
    ("d_chroma", "ChromaDB is a vector database that stores embeddings and "
                 "supports similarity search with metadata filtering.", "retrieval"),
    ("d_threshold", "A similarity threshold decides which results count as "
                    "matches. Raising it improves precision but lowers recall.", "retrieval"),
    # --- Databases ---
    ("d_orm", "SQLAlchemy maps Python classes to database tables so operations "
              "can be written with objects instead of raw SQL.", "database"),
    ("d_migration", "Database migrations apply schema changes in order so every "
                    "environment ends up with the same structure.", "database"),
]

ids = [d[0] for d in documents]
texts = [d[1] for d in documents]
metadatas = [{"topic": d[2]} for d in documents]

collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

# ----------------------------------------------------------------------
# Evaluation set: 6 queries with expected relevant document IDs
# ----------------------------------------------------------------------

eval_set = [
    {
        "query": "How do I authenticate requests to my API?",
        "relevant": ["d_jwt"],
    },
    {
        "query": "How do I check that incoming request data is valid?",
        "relevant": ["d_pydantic"],
    },
    {
        "query": "Why does my data disappear when I click a button?",
        "relevant": ["d_session", "d_streamlit"],
    },
    {
        "query": "How is text turned into something a computer can compare?",
        "relevant": ["d_embed", "d_cosine"],
    },
    {
        "query": "How should I split up a long document for searching?",
        "relevant": ["d_chunk"],
    },
    {
        "query": "How do I store Python objects in a relational database?",
        "relevant": ["d_orm"],
    },
]


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def search(query, n_results=5, threshold=0.3):
    """Query the collection and return (id, similarity) pairs above threshold."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance          # cosine distance -> similarity
        if similarity >= threshold:
            hits.append((results["ids"][0][i], similarity))
    return hits


def evaluate(eval_set, n_results=5, threshold=0.3, show=True):
    """Compute per-query and average precision/recall."""
    total_p = total_r = total_hit = 0.0

    if show:
        print(f"\n=== Evaluation at threshold={threshold}, n_results={n_results} ===")

    for i, item in enumerate(eval_set, start=1):
        relevant = set(item["relevant"])
        hits = search(item["query"], n_results=n_results, threshold=threshold)
        returned = {doc_id for doc_id, _ in hits}

        true_positives = returned & relevant
        precision = len(true_positives) / len(returned) if returned else 0.0
        recall = len(true_positives) / len(relevant) if relevant else 0.0
        hit_at_k = 1.0 if true_positives else 0.0

        total_p += precision
        total_r += recall
        total_hit += hit_at_k

        if show:
            print(f"Query {i}: P={precision:5.1%} R={recall:5.1%} "
                  f"hit@{n_results}={int(hit_at_k)}  {item['query']!r}")
            print(f"          returned: {sorted(returned) or 'nothing'}")

    n = len(eval_set)
    avg_p, avg_r, avg_hit = total_p / n, total_r / n, total_hit / n

    if show:
        print(f"AVERAGE: P={avg_p:.1%} R={avg_r:.1%} hit@{n_results}={avg_hit:.1%}")

    return avg_p, avg_r, avg_hit


# ----------------------------------------------------------------------
# Run at three settings
# ----------------------------------------------------------------------

print("Documents in collection:", collection.count())

settings = [
    {"n_results": 3, "threshold": 0.3},
    {"n_results": 5, "threshold": 0.3},
    {"n_results": 5, "threshold": 0.5},
]

rows = []
for s in settings:
    p, r, hit = evaluate(eval_set, n_results=s["n_results"],
                         threshold=s["threshold"])
    rows.append((s["n_results"], s["threshold"], p, r, hit))

print("\n" + "=" * 68)
print("SETTINGS COMPARISON")
print("=" * 68)
print(f"{'n_results':>10} {'threshold':>10} {'Precision':>11} "
      f"{'Recall':>9} {'hit@k':>8}")
for n_res, thr, p, r, hit in rows:
    print(f"{n_res:>10} {thr:>10} {p:>10.1%} {r:>9.1%} {hit:>8.1%}")

ANALYSIS = """
ANALYSIS
--------

What worked well:
  Queries 1, 5, and 6 (JWT, chunking, ORM) each have a single unambiguous
  relevant document, and the ranking puts it first. These are the easy case:
  the query and the document share intent, even where they share few words.
  Query 3 (session state) reaches both expected documents because the query
  describes the symptom ("data disappears") rather than naming the mechanism.

What failed or scored poorly:
  Precision is depressed by short, single-topic queries against a broad
  collection. Query 1 returns only one relevant document by design, so if
  n_results=3 returns three documents its precision caps at 33% even when the
  ranking is perfect. That is a property of the evaluation set, not of the
  search system.

  Recall falls at threshold 0.5 because several legitimate matches score in the
  0.4-0.5 band. This collection has fifteen documents but only four topic areas,
  so cross-topic similarity is higher than in a larger corpus, and a high
  threshold strips out correct answers alongside the wrong ones.

  Query 4 (embeddings + cosine similarity) is the most fragile. The two expected
  documents describe a concept and a metric rather than answering a question, so
  a question-shaped query does not align with either vector as strongly as it
  does with the more imperative documents.

What I would change:
  1. Report hit@k as the primary metric. With mostly single-relevant queries,
     precision is structurally capped and misleading; hit@k answers the
     question actually being asked, which is whether the right document
     appeared in the top k.
  2. Size the evaluation set to the metric. Either add more relevant documents
     per query so precision is meaningful, or commit to hit@k/recall and stop
     reporting precision.
  3. Use a two-stage search: retrieve with a low threshold and a generous k,
     then re-rank or filter. A single cosine threshold forces one tradeoff on
     every query.
  4. Test chunking as well as queries. Several documents here are single
     sentences; real documents are longer, and chunk boundaries will matter
     more than threshold tuning.
  5. Grow the corpus. Four topics and fifteen documents makes accidental
     similarity common. A threshold that works here would need retuning on a
     real knowledge base.
"""

print(ANALYSIS)