

import os


def mock_response(prompt):
    """Simulates how prompt quality affects response quality."""
    prompt_lower = prompt.lower()

    has_role = any(word in prompt_lower for word in ["you are", "act as", "your role"])
    has_constraints = any(word in prompt_lower for word in
        ["under 100", "in 3 sentences", "as a table", "as json", "format", "bullet"])
    has_examples = "input:" in prompt_lower or "example:" in prompt_lower

    quality_score = sum([has_role, has_constraints, has_examples])

    if quality_score == 0:
        return ("[MOCK - Vague prompt detected]\n"
                "Embeddings are a way to represent data as numbers. "
                "They are used in machine learning and NLP. "
                "There are many types of embeddings.\n"
                "(This generic response demonstrates what happens with vague prompts.)")
    elif quality_score == 1:
        return ("[MOCK - Decent prompt]\n"
                "Embeddings convert text into numerical vectors that capture semantic "
                "meaning. Think of it like a GPS coordinate for meaning - similar "
                "ideas get similar coordinates. This is how semantic search works: "
                "instead of matching keywords, you compare meaning vectors.\n"
                "(Better - the prompt gave some direction.)")
    else:
        return ("[MOCK - Well-engineered prompt]\n"
                "| Feature | Keyword Search | Semantic Search |\n"
                "|---------|---------------|-----------------|\n"
                "| Matching | Exact words | Meaning/intent |\n"
                "| Handles synonyms | No | Yes |\n"
                "| Requires | Word overlap | Embedding model |\n\n"
                "Think of keyword search like a librarian who only looks at book "
                "titles. Semantic search is a librarian who actually read every book "
                "and can recommend the right one even if you describe it differently.\n"
                "(Excellent - role + constraints + format produced focused output.)")


def get_response(prompt):
    """Try real API, fall back to mock."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content
    else:
        return mock_response(prompt)


print("=" * 60)
print("EXPERIMENT 1: The Power of Framing")
print("=" * 60)

prompt_no_frame = "Explain embeddings."
prompt_with_frame = (
    "You are a patient programming instructor teaching career changers "
    "who know Python and FastAPI but are new to AI concepts. "
    "Explain what embeddings are in the context of semantic search. "
    "Use a real-world analogy before any technical terms. "
    "Keep it under 100 words."
)

print(f"\nPrompt (no framing): {prompt_no_frame!r}")
print(f"Response:\n{get_response(prompt_no_frame)}\n")

print(f"Prompt (with framing): {prompt_with_frame[:80]!r}...")
print(f"Response:\n{get_response(prompt_with_frame)}\n")

print("=" * 60)
print("EXPERIMENT 2: Few-Shot Examples")
print("=" * 60)

prompt_few_shot = """Convert these plain descriptions to ChromaDB metadata dictionaries:

Input: "This is from the FastAPI module about authentication"
Output: {"module": "5", "topic": "authentication", "subtopic": "JWT"}

Input: "A Streamlit lesson about session state management"
Output: {"module": "6", "topic": "frontend", "subtopic": "state_management"}

Input: "Explains how cosine similarity works for comparing embeddings"
Output:"""

print(f"\nFew-shot prompt (showing pattern):\n{prompt_few_shot}")
print(f"\nResponse:\n{get_response(prompt_few_shot)}")

print("\n" + "=" * 60)
print("KEY TAKEAWAY: Same question, dramatically different results.")
print("The prompt IS the product.")
print("=" * 60)