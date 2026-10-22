from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# A multi-topic document covering concepts from the course.
document = """
FastAPI is a Python framework for building web APIs. An API provides endpoints that clients access with HTTP requests. Common HTTP methods include GET for retrieving data, POST for creating resources, PUT or PATCH for updating resources, and DELETE for removing resources. Status codes tell the client what happened. A successful request might return 200, while a newly created resource often returns 201. Client errors include 400 for a bad request and 404 when a resource cannot be found.

Pydantic models describe the expected structure of request and response data. A model defines fields, data types, optional values, and validation rules. FastAPI uses these models to validate incoming JSON automatically. When submitted data does not match the model, FastAPI returns a detailed 422 response. Pydantic also converts compatible values to the correct Python types and helps produce consistent API responses.

FastAPI generates interactive documentation from endpoint definitions and type hints. Swagger UI is normally available at the /docs path, while ReDoc is available at /redoc. Developers can use this documentation to inspect endpoints, understand required parameters, and send test requests. Because the documentation is generated from the application code, it stays synchronized as endpoints change.

Asynchronous programming helps an application handle multiple operations efficiently. A function declared with async def can pause with await while it waits for work such as a database query or an external HTTP request. During that pause, the server can process other requests instead of blocking. Async code is most useful for input and output operations, but it does not automatically make CPU-intensive work faster.

Authentication verifies a user's identity, while authorization determines what that user is permitted to do. JSON Web Tokens are often used for stateless API authentication. A login endpoint checks the user's credentials and returns a signed token. The client sends that token in the Authorization header when accessing protected endpoints. The server verifies the token's signature and expiration before allowing the request.

SQLAlchemy is a Python toolkit for communicating with relational databases. Its object-relational mapping features connect Python classes to database tables. Application code can create, retrieve, update, and delete records through Python objects rather than writing every SQL statement directly. Database sessions track changes and transactions ensure that related operations either succeed together or are rolled back.

Testing confirms that an application behaves as expected. FastAPI's TestClient can simulate HTTP requests without running a separate server. Pytest discovers test functions, executes them, and reports failures. Tests can verify response status codes, JSON content, validation errors, and authentication behavior. Fixtures provide reusable setup data, while mocking can isolate the application from external services.

Git records changes to project files as commits. Developers stage changes with git add, create a commit with git commit, and upload commits to a remote repository with git push. A .gitignore file prevents generated files, virtual environments, secrets, and cache directories from entering the repository. Descriptive commit messages make project history easier to understand.

Embeddings convert text into numerical vectors that represent semantic meaning. Sentences with related meanings tend to have vectors pointing in similar directions. Cosine similarity measures the angle between vectors and can rank documents against a search query. Semantic search can therefore retrieve relevant passages even when the query and passage use different vocabulary.

Chunking divides long documents into smaller pieces before embedding them. Fixed-size chunking is predictable and works even when a document has poor formatting, but it can split sentences or ideas at arbitrary boundaries. Overlap preserves some context between adjacent fixed chunks. Paragraph chunking tends to preserve complete ideas in well-structured documents, but very long or inconsistent paragraphs can produce uneven chunks. The best strategy depends on the structure of the source material and the kinds of questions users ask.
""".strip()


def fixed_chunks(text, size=300, overlap=50):
    """Split text into fixed-size chunks with character overlap."""
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start += size - overlap

    return chunks


def paragraph_chunks(text):
    """Split text wherever a blank line separates paragraphs."""
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def search(query, chunks, chunk_embeddings, top_k=2):
    """Return the highest-scoring chunks for a query."""
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        (float(score), chunks[index])
        for index, score in ranked[:top_k]
    ]


fixed = fixed_chunks(document)
paragraphs = paragraph_chunks(document)

fixed_embeddings = model.encode(fixed)
paragraph_embeddings = model.encode(paragraphs)

queries = [
    "How does the API reject data that has the wrong format?",
    "How can users access protected endpoints without server sessions?",
    "Why might paragraphs be better than cutting text by character count?",
]

print("=== Chunk Statistics ===")
print(
    f"Fixed-size: {len(fixed)} chunks, "
    f"average {sum(map(len, fixed)) // len(fixed)} characters"
)
print(
    f"Paragraph: {len(paragraphs)} chunks, "
    f"average {sum(map(len, paragraphs)) // len(paragraphs)} characters"
)

for query in queries:
    print(f"\n{'=' * 78}")
    print(f"Query: {query!r}")

    strategy_results = [
        ("Fixed-size chunks", search(query, fixed, fixed_embeddings)),
        ("Paragraph chunks", search(query, paragraphs, paragraph_embeddings)),
    ]

    for strategy_name, results in strategy_results:
        print(f"\n  {strategy_name} - top 2 results:")

        for rank, (score, chunk) in enumerate(results, start=1):
            clean_chunk = " ".join(chunk.split())
            print(f"    {rank}. [{score:.4f}] {clean_chunk}")

print(f"\n{'=' * 78}")
print("Comparison:")
print(
    "Paragraph chunking performed better overall for this document because each "
    "paragraph focuses on one course topic and preserves a complete explanation. "
    "Its results were generally easier to interpret and contained enough context "
    "to answer the queries. Fixed-size chunking still found relevant text, but some "
    "chunks began or ended in the middle of a sentence or combined parts of adjacent "
    "topics. Fixed-size chunks would be more useful for poorly formatted documents "
    "without dependable paragraph breaks. The scores and retrieved passages above "
    "should be reviewed together, since the highest numerical score does not always "
    "produce the most complete answer."
)
