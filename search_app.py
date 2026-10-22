import os

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")


@st.cache_resource
def get_collection():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path="./search_db")
    return client.get_or_create_collection(
        name="course_docs",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


collection = get_collection()


def load_and_chunk(directory):
    """Load text files and split into paragraph chunks."""
    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            chunks.append({
                "text": para,
                "source": filename,
                "chunk_id": f"{filename}_{i}",
                "chunk_index": i,
            })
    return chunks


def relevance_badge(distance):
    if distance < 0.5:
        return "🟢 High", "#1a7f37"
    if distance < 1.0:
        return "🟡 Medium", "#9a6700"
    return "🔴 Low", "#a40e26"


def run_search(query, n_results, sources):
    """Query ChromaDB, applying a source filter when files are selected."""
    where = {"source": {"$in": sources}} if sources else None

    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )


with st.sidebar:
    st.title("📁 Document Manager")

    if st.button("🔄 Re-index Documents"):
        chunks = load_and_chunk("docs")
        if chunks:
            collection.upsert(
                documents=[c["text"] for c in chunks],
                metadatas=[
                    {"source": c["source"], "chunk_index": str(c["chunk_index"])}
                    for c in chunks
                ],
                ids=[c["chunk_id"] for c in chunks],
            )
            st.success(
                f"Indexed {len(chunks)} chunks from "
                f"{len(set(c['source'] for c in chunks))} files"
            )
        else:
            st.warning("No .txt or .md files found in docs/ folder")

    st.metric("Chunks in DB", collection.count())

    # Available sources come from the docs folder on disk.
    all_sources = sorted(
        f for f in os.listdir("docs")
        if f.endswith((".txt", ".md"))
    )
    st.metric("Source files", len(all_sources))

    st.divider()

    selected_sources = st.multiselect(
        "Search only in these files",
        options=all_sources,
        default=[],
        help="Leave empty to search every file.",
    )

    n_results = st.slider("Results to show", 1, 10, 5)


st.title("🔍 Semantic Search")
st.write("Search your course documents by meaning, not just keywords.")


# The "similar to this" button stores its text here, then triggers a fresh search.
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

query = st.text_input(
    "Enter your search query",
    value=st.session_state.query_text,
    placeholder="How does authentication work?",
)

if query and collection.count() > 0:
    scope = selected_sources or all_sources

    # Only count the chunks that are actually in scope.
    total_in_scope = 0
    for source in scope:
        total_in_scope += collection.get(where={"source": source})["ids"].__len__()

    results = run_search(query, n_results=min(n_results, total_in_scope), sources=selected_sources)

    shown = len(results["documents"][0])
    st.caption(
        f"Showing {shown} of {total_in_scope} total documents"
        + (f" (filtered to {len(selected_sources)} file(s))" if selected_sources else "")
    )

    st.subheader("Results")

    for i in range(shown):
        doc = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        chunk_id = results["ids"][0][i]

        label, color = relevance_badge(distance)

        with st.container():
            col_meta, col_score = st.columns([3, 1])
            with col_meta:
                st.markdown(
                    f"**{metadata['source']}** — chunk {metadata['chunk_index']}"
                )
            with col_score:
                st.markdown(
                    f"<span style='background-color:{color};color:white;"
                    f"padding:2px 8px;border-radius:6px;font-size:0.85em;'>"
                    f"{label} · {distance:.3f}</span>",
                    unsafe_allow_html=True,
                )

            preview = " ".join(doc.split())
            if len(preview) > 150:
                preview = preview[:150].rstrip() + "…"
            st.write(preview)

            with st.expander("Show full text"):
                st.write(doc)

            if st.button("🔎 Similar to this", key=f"sim_{chunk_id}"):
                st.session_state.query_text = doc
                st.rerun()

            st.divider()

elif collection.count() == 0:
    st.info("👈 Click 'Re-index Documents' in the sidebar to load your documents first.")
