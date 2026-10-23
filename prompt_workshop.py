import json
import os


def mock_response(prompt, system=None):
    """Stand-in for a real model. Detects prompt quality by keyword."""
    text = prompt.lower()
    if system:
        text = system.lower() + " " + text

    has_role = any(w in text for w in ["you are", "act as", "your role", "assistant"])
    has_constraints = any(w in text for w in
        ["under 150", "under 100", "in 3 sentences", "as json", "as a table",
         "format", "bullet", "respond only", "return only"])
    has_examples = "input:" in text or "example:" in text
    has_context = "context:" in text or "provided context" in text

    score = sum([has_role, has_constraints, has_examples, has_context])

    if "json" in text:
        if has_examples:
            return json.dumps([
                {"title": "Write project README", "priority": "high", "status": "todo"},
                {"title": "Fix login bug", "priority": "high", "status": "in_progress"},
                {"title": "Update dependencies", "priority": "low", "status": "todo"},
            ], indent=2)
        return ('{"tasks": [\n  {"title": "Write project README", "priority": "medium"},\n'
                '  {"title": "Fix login bug", "priority": "medium"}\n]}\n'
                "(MOCK - no example given, so the shape is guessed.)")

    if has_context or has_role and has_constraints:
        return ("[MOCK - Well-engineered prompt]\n"
                "Based on streamlit_guide.txt: st.session_state is a dictionary that "
                "survives script re-runs. Because Streamlit re-executes your whole "
                "script on every widget interaction, ordinary variables reset each "
                "time. session_state persists across those re-runs, which is how you "
                "keep a selected value or accumulated data alive between interactions.\n"
                "(Source: streamlit_guide.txt)")

    if score == 1:
        return ("[MOCK - Decent prompt]\n"
                "st.session_state stores values that stay available across interactions. "
                "It is commonly used when variables would otherwise reset.\n"
                "(Better - the prompt gave some direction.)")

    return ("[MOCK - Vague prompt detected]\n"
            "Streamlit is a Python library for building web applications. "
            "It has many features for making interactive apps. "
            "session_state is one of those features.\n"
            "(This generic response demonstrates what happens with vague prompts.)")


def get_response(prompt, system=None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        import openai
        client = openai.OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
        )
        return response.choices[0].message.content
    return mock_response(prompt, system=system)


def compare(task_title, bad_prompt, good_prompt, system=None):
    print("\n" + "=" * 70)
    print(task_title)
    print("=" * 70)

    print("\n--- BAD PROMPT ---")
    print(bad_prompt)
    print("\nResponse:")
    print(get_response(bad_prompt, system=system))

    print("\n--- GOOD PROMPT ---")
    print(good_prompt)
    print("\nResponse:")
    print(get_response(good_prompt, system=system))


# ----------------------------------------------------------------------
# TASK 1 - Code explanation
# ----------------------------------------------------------------------
compare(
    "TASK 1: Explain st.session_state",
    "What does st.session_state do?",
    (
        "You are a Streamlit instructor explaining concepts to developers who know "
        "Python but are new to Streamlit's execution model. "
        "Explain what st.session_state does and why it is needed. "
        "Start by naming the problem it solves, then describe how it solves it. "
        "Include one short code snippet showing the initialization pattern. "
        "Keep it in 3 sentences before the code."
    ),
)

# ----------------------------------------------------------------------
# TASK 2 - Data formatting
# ----------------------------------------------------------------------
compare(
    "TASK 2: Convert a task list to JSON",
    (
        "Turn this into JSON:\n"
        "write the project readme, fix the login bug, update dependencies"
    ),
    (
        "You are a data formatter. Convert each task in the input into a JSON object "
        "with exactly three fields: title, priority, and status. "
        "Use priority values of high, medium, or low, and status values of todo, "
        "in_progress, or done. Infer priority from the phrasing if it is not stated. "
        "Respond only with a valid JSON array and nothing else.\n\n"
        "Input: \"deploy the staging build, urgently fix the payment bug, "
        "someday reorganize the test folder\"\n"
        "Output: [\n"
        "  {\"title\": \"Deploy the staging build\", \"priority\": \"medium\", "
        "\"status\": \"todo\"},\n"
        "  {\"title\": \"Fix the payment bug\", \"priority\": \"high\", "
        "\"status\": \"todo\"}\n"
        "]\n\n"
        "Input: \"write the project readme, fix the login bug, update dependencies\"\n"
        "Output:"
    ),
)

# ----------------------------------------------------------------------
# TASK 3 - System prompt design
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Course Study Assistant for this course's notes.

Rules:
1. Answer using ONLY the context documents provided with the question. Do not use
   outside knowledge, and do not guess or fill gaps from memory.
2. If the provided context does not contain the answer, say so plainly: "The
   course notes provided don't cover that." Then stop. Do not attempt a partial
   answer from general knowledge.
3. Keep every answer under 150 words.
4. End every answer with a single line in this exact form:
   Source: <filename>
   Name the one source document the answer came from. If the answer drew on more
   than one, list them comma-separated.
5. Use plain prose. No preamble, no restating the question, no pleasantries.
"""

compare(
    "TASK 3: Course Study Assistant system prompt",
    "Answer questions about the course.",
    SYSTEM_PROMPT.strip(),
    system=SYSTEM_PROMPT,
)

print("\n" + "=" * 70)
print("SYSTEM PROMPT (final, ready for Module 8's RAG pipeline):")
print("=" * 70)
print(SYSTEM_PROMPT)