from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base: 4 topics, 4 sentences each
docs = [
    # Python development
    "Deploy FastAPI with uvicorn and Docker",
    "Python web frameworks comparison",
    "Setting up a virtual environment",
    "Writing automated tests with pytest",
    # Version control / tooling
    "Git basics for version control",
    "SQL database migration strategies",
    # Cooking
    "The best homemade pasta recipe with fresh tomatoes",
    "How to make pizza dough from scratch",
    "Slow-roasting vegetables brings out their sweetness",
    "Choosing the right knife for kitchen prep",
    # Space
    "NASA's latest Mars rover discovered water ice",
    "SpaceX launched a new satellite into orbit",
    "How orbital mechanics determine satellite trajectories",
    "The James Webb telescope observes infrared light",
    # Music
    "Modes and scales in jazz improvisation",
    "Mixing vocals with compression and reverb",
    "How chord progressions create tension and release",
    "Recording drums with a minimal microphone setup",
]

queries = [
    "How do I deploy my Python app?",          # clearly Python
    "What should I cook for dinner?",          # clearly cooking
    "How do I get better at playing music?",   # clearly music
    "How do I put my project online?",         # ambiguous: Python + Git
    "What's new in astronomy?",                # ambiguous: space
]

THRESHOLDS = [0.3, 0.5, 0.7]

doc_embeddings = model.encode(docs)

for query in queries:
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)

    print(f"\nQuery: {query!r}")

    for t in THRESHOLDS:
        passing = [(idx, float(score)) for idx, score in ranked if float(score) >= t]
        print(f"\n  Threshold {t}: {len(passing)} results")
        if not passing:
            print("    (none)")
        for idx, score in passing:
            print(f"    [{score:.2f}] {docs[idx]}")

    # What a strict threshold costs you, relative to the loosest one
    loosest = {idx for idx, s in ranked if float(s) >= 0.3}
    strictest = {idx for idx, s in ranked if float(s) >= 0.7}
    missed = loosest - strictest

    print(f"\n  Missed at 0.7 but kept at 0.3: {len(missed)}")
    for idx in sorted(missed, key=lambda i: -float(scores[i])):
        print(f"    [{float(scores[idx]):.2f}] {docs[idx]}")