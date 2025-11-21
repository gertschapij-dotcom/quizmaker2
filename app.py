"""
Streamlit Quiz App — Gemini (gemini-2.5-flash-lite) with API key input in the UI.

How to use:
- Paste your Google API key into the "Google API Key" field in the sidebar (password field).
- Upload a text file or enter a subject, choose difficulty/number/type, then press "Generate Quiz".
- The app will call gemini-2.5-flash-lite using the provided key. If generation/parsing fails, it falls back
  to a built-in self-contained generator and shows diagnostics in the sidebar.

Dependencies:
  pip install streamlit requests
Run:
  streamlit run streamlit_app.py
"""
from dataclasses import dataclass
import json
import random
import re
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

# Fixed model name
GOOGLE_MODEL = "gemini-2.5-flash-lite"

# -----------------------
# Data structures
# -----------------------
@dataclass
class QuizQuestion:
    id: int
    type: str  # "multiple_choice" or "open"
    prompt: str
    choices: Optional[List[str]]
    answer: str
    explanation: str

# -----------------------
# Helpers
# -----------------------
def extract_json(text: str) -> Optional[str]:
    """Try to extract first valid JSON object/array from text."""
    if not text:
        return None
    candidates = []
    # Find {...} spans
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            stack.append(i)
        elif ch == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:i+1])
                start = None
    # Find [...] spans
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '[':
            if start is None:
                start = i
            stack.append(i)
        elif ch == ']' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:i+1])
                start = None
    for c in candidates:
        try:
            json.loads(c)
            return c
        except Exception:
            continue
    # Last resort: try whole text
    try:
        json.loads(text)
        return text
    except Exception:
        return None

def simple_fallback_quiz(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> List[QuizQuestion]:
    """A small deterministic fallback quiz generator."""
    source = (context_text or subject or "General knowledge").strip()
    sentences = re.split(r'(?<=[.!?])\s+', source)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    qs: List[QuizQuestion] = []
    for i in range(n_questions):
        sent = sentences[i] if i < len(sentences) else (subject or "This subject") + " overview."
        q_prompt = sent.rstrip('.')
        if q_type == "multiple_choice":
            words = re.findall(r"\w+", sent)
            key = words[-1] if words else "concept"
            choices = [key, key + "s", "another concept", "not related"]
            random.shuffle(choices)
            answer = choices[0]
            explanation = f"Important term: {key}. Source: {sent}"
            qs.append(QuizQuestion(id=i+1, type="multiple_choice", prompt=f"What is an important fact in this statement: {q_prompt}?", choices=choices, answer=answer, explanation=explanation))
        else:
            answer = sent
            explanation = f"Key points: {sent}"
            qs.append(QuizQuestion(id=i+1, type="open", prompt=f"What is an important fact in this statement: {q_prompt}?", choices=None, answer=answer, explanation=explanation))
    return qs

def simple_fallback_grade(reference: str, student_answer: str) -> Dict[str, Any]:
    ref_words = set(re.findall(r"\w+", reference.lower()))
    stud_words = set(re.findall(r"\w+", student_answer.lower()))
    common = ref_words & stud_words
    score = int(100 * len(common) / len(ref_words)) if ref_words else 0
    feedback = f"Fallback grading: {len(common)} overlapping keywords. Reference: {reference}"
    return {"score": score, "feedback": feedback}

# -----------------------
# Prompt builders
# -----------------------
def build_generation_prompt(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> str:
    source = context_text if context_text else f"Subject: {subject}"
    prompt = f"""You are a helpful quiz writer. Create a quiz of {n_questions} question(s) in JSON format ONLY.
Context:
\"\"\"{source}\"\"\"

Difficulty: {difficulty}
Question type preference: {q_type}

IMPORTANT: Output ONLY valid JSON and nothing else. The top-level structure must be:
{{ "questions": [ ... ] }}

Each question must include:
- "id": integer
- "type": "multiple_choice" or "open"
- "prompt": question text
- "choices": array of strings (for multiple_choice) or null/omit for open
- "answer": for multiple_choice the exact correct choice text; for open a short model answer/keypoints
- "explanation": detailed explanation to be shown after answering

Do not add commentary, labels, or any non-JSON text."""
    return prompt

def build_grading_prompt(question: str, reference: str, student_answer: str, difficulty: str) -> str:
    return f"""You are an objective grader. Grade the student's answer from 0 to 100 based on correctness and completeness relative to the reference answer.
Question: {question}
Reference / expected answer: {reference}
Student answer: {student_answer}
Difficulty: {difficulty}

Return ONLY valid JSON like: {{ "score": <int 0-100>, "feedback": "<detailed feedback explaining strengths, weaknesses, and how to improve>" }}"""

# -----------------------
# Google Generative calls (tries v1 and v1beta2)
# -----------------------
def call_google_generate(model: str, api_key: str, prompt_text: str, max_output_tokens: int = 1024, temperature: float = 0.0, timeout: int = 60) -> Optional[str]:
    """Call Google Generative Language API using the provided API key. Writes diagnostics to sidebar."""
    if not api_key:
        return None

    model_path = model if model.startswith("models/") else f"models/{model}"
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/{model_path}:generateText?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta2/{model_path}:generateText?key={api_key}"
    ]

    last_status = None
    last_text = None

    for url in endpoints:
        try:
            headers = {"Content-Type": "application/json"}
            body = {"prompt": {"text": prompt_text}, "temperature": temperature, "maxOutputTokens": max_output_tokens}
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            status = resp.status_code
            text = resp.text
            last_status = status
            last_text = text

            # show diagnostics in sidebar (helpful)
            try:
                st.sidebar.markdown(f"**Google endpoint tried:** `{url}`")
                st.sidebar.markdown(f"**HTTP status:** `{status}`")
                st.sidebar.text_area("Raw Google response (truncated)", value=(text[:4000] + ("...[truncated]" if len(text) > 4000 else "")), height=220)
            except Exception:
                # fallback: print to console
                print("Google endpoint tried:", url)
                print("Status:", status)
                print("Body:", text[:2000])

            if not resp.ok:
                continue

            data = resp.json()
            if isinstance(data, dict):
                if data.get("error"):
                    last_text = json.dumps(data)
                    continue
                candidates = data.get("candidates") or data.get("responses") or []
                if isinstance(candidates, list) and len(candidates) > 0:
                    first = candidates[0]
                    if isinstance(first, dict):
                        for k in ("content", "output", "text", "generated_text"):
                            if first.get(k):
                                return first.get(k)
                    if isinstance(first, str) and first.strip():
                        return first
            # fallback: find first string anywhere
            def find_string(o):
                if isinstance(o, str):
                    return o
                if isinstance(o, dict):
                    for v in o.values():
                        s = find_string(v)
                        if s:
                            return s
                if isinstance(o, list):
                    for e in o:
                        s = find_string(e)
                        if s:
                            return s
                return None
            found = find_string(data)
            if found:
                return found
            return text
        except Exception as e:
            last_status = "exception"
            last_text = str(e)
            try:
                st.sidebar.error(f"Google call exception: {e}")
            except Exception:
                print("Google call exception:", e)
            continue

    # all endpoints tried and failed
    try:
        st.sidebar.error(f"All Google endpoints failed. Last status: {last_status}")
        st.sidebar.text_area("Last Google response body (debug)", value=(last_text or "no body"), height=220)
    except Exception:
        print("All Google endpoints failed. Last status:", last_status)
        print("Last body:", last_text)
    return None

def generate_quiz_with_google(api_key: str, context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> Optional[List[QuizQuestion]]:
    if not api_key:
        return None
    prompt = build_generation_prompt(context_text, subject, difficulty, n_questions, q_type)
    text = call_google_generate(GOOGLE_MODEL, api_key, prompt, max_output_tokens=1024, temperature=0.0)
    if not text:
        return None
    json_str = extract_json(text) or text
    try:
        parsed = json.loads(json_str)
        qs: List[QuizQuestion] = []
        for q in parsed.get("questions", []):
            qid = int(q.get("id", len(qs)+1))
            qtype = q.get("type", "open")
            prompt_text = q.get("prompt", "").strip()
            choices = q.get("choices", None)
            answer = q.get("answer", "")
            explanation = q.get("explanation", "")
            qs.append(QuizQuestion(id=qid, type=qtype, prompt=prompt_text, choices=choices, answer=answer, explanation=explanation))
        if len(qs) > n_questions:
            qs = qs[:n_questions]
        return qs
    except Exception:
        try:
            st.sidebar.text_area("Failed to parse JSON from model output (raw):", value=text, height=300)
        except Exception:
            print("Failed to parse JSON from model output. Raw text:", text)
        return None

def grade_open_answer_with_google(api_key: str, question: str, reference: str, student_answer: str, difficulty: str) -> Dict[str, Any]:
    if not api_key:
        return simple_fallback_grade(reference, student_answer)
    prompt = build_grading_prompt(question, reference, student_answer, difficulty)
    text = call_google_generate(GOOGLE_MODEL, api_key, prompt, max_output_tokens=400, temperature=0.0)
    if not text:
        return {"score": 0, "feedback": "Grading failed (no response from Google)."}
    json_str = extract_json(text) or text
    try:
        parsed = json.loads(json_str)
        score = int(parsed.get("score", 0))
        feedback = parsed.get("feedback", "")
        return {"score": score, "feedback": feedback}
    except Exception:
        return simple_fallback_grade(reference, student_answer)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Quiz Generator — Gemini (UI key)", layout="wide")
st.title("Quiz Generator — Gemini (gemini-2.5-flash-lite)")

st.sidebar.header("Google API Key (paste here)")
st.sidebar.markdown("Paste your Google API key below. It will be used for calls to the Gemini model. If missing or invalid the app will fall back to a local fallback generator.")
google_api_key = st.sidebar.text_input("Google API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("Quiz Source")
source_choice = st.sidebar.radio("Create quiz from:", ("Upload text file", "Subject / short description"))
uploaded_text = ""
subject_line = ""
if source_choice == "Upload text file":
    uploaded_file = st.sidebar.file_uploader("Upload a large text file (txt, md). For PDFs, copy content into a .txt.", type=["txt", "md"])
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                try:
                    uploaded_text = raw.decode("utf-8")
                except Exception:
                    uploaded_text = raw.decode("latin-1", errors="ignore")
            else:
                uploaded_text = str(raw)
            st.sidebar.success("File loaded.")
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
else:
    subject_line = st.sidebar.text_input("Subject or short description", value="Photosynthesis and how plants convert light into energy")

st.sidebar.header("Quiz Settings")
difficulty = st.sidebar.selectbox("Difficulty", ["easy", "medium", "hard"])
n_questions = st.sidebar.slider("Number of questions", 1, 20, 5)
q_type = st.sidebar.selectbox("Question type", ["multiple_choice", "open", "mix"])
randomize = st.sidebar.checkbox("Randomize question order", value=True)

# Test button to run a quick verification call
if st.sidebar.button("Test API Key"):
    if not google_api_key:
        st.sidebar.error("No API key provided in the UI. Paste the key and try again.")
    else:
        test_prompt = 'Respond with only the JSON: {"ok": true}'
        out = call_google_generate(GOOGLE_MODEL, google_api_key, test_prompt, max_output_tokens=64, temperature=0.0)
        if out:
            js = extract_json(out)
            if js:
                try:
                    parsed = json.loads(js)
                    if isinstance(parsed, dict) and parsed.get("ok") is True:
                        st.sidebar.success("Test succeeded: model returned expected JSON.")
                    else:
                        st.sidebar.warning("Test returned content but not the exact JSON requested. See raw response above.")
                except Exception:
                    st.sidebar.warning("Received response but failed to parse JSON. See raw response above.")
            else:
                st.sidebar.info("Received non-JSON response from model. See raw response above.")
        else:
            st.sidebar.error("No response from Google. See the raw response diagnostics in the sidebar.")

# Generate quiz button
generate_button = st.sidebar.button("Generate Quiz")

# Session state
if "quiz" not in st.session_state:
    st.session_state["quiz"] = None
if "answers" not in st.session_state:
    st.session_state["answers"] = {}
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None

if generate_button:
    st.session_state["answers"] = {}
    st.session_state["feedback"] = None
    context_text = uploaded_text.strip() if uploaded_text else ""
    subject = subject_line.strip() if subject_line else ""
    if not context_text and not subject:
        st.sidebar.error("Please provide an uploaded text file or a subject line.")
    else:
        with st.spinner("Generating quiz..."):
            qs: Optional[List[QuizQuestion]] = None
            if google_api_key:
                qs = generate_quiz_with_google(google_api_key, context_text, subject, difficulty, n_questions, q_type)
                if qs is None:
                    st.warning("Google generation failed or returned invalid output. Falling back to internal generator.")
            if qs is None:
                qs = simple_fallback_quiz(context_text, subject, difficulty, n_questions, q_type if q_type != "mix" else "mixed")
            if randomize:
                random.shuffle(qs)
            st.session_state["quiz"] = qs
            st.success(f"Generated {len(qs)} question(s).")

# Display quiz and collect answers
quiz = st.session_state.get("quiz")
if quiz:
    st.header("Quiz")
    form = st.form(key="quiz_form")
    answers: Dict[str, str] = {}
    for q in quiz:
        with st.expander(f"Question {q.id}: {q.type}", expanded=True):
            st.write(q.prompt)
            if q.type == "multiple_choice":
                choices = q.choices or ["A", "B", "C", "D"]
                selected = form.radio(f"Select answer for Q{q.id}", choices, key=f"q_{q.id}")
                answers[str(q.id)] = selected
            else:
                txt = form.text_area(f"Your answer for Q{q.id}", key=f"q_{q.id}", height=120)
                answers[str(q.id)] = txt
    submit = form.form_submit_button("Submit Answers")
    if submit:
        st.session_state["answers"] = answers
        total_score = 0
        max_score = 0
        feedback_list = []
        with st.spinner("Grading..."):
            for q in quiz:
                user_ans = answers.get(str(q.id), "")
                if q.type == "multiple_choice":
                    max_score += 1
                    is_correct = False
                    if isinstance(q.answer, str) and user_ans.strip().lower() == q.answer.strip().lower():
                        is_correct = True
                    if not is_correct and q.choices:
                        try:
                            idx = int(q.answer) - 1
                            if 0 <= idx < len(q.choices) and q.choices[idx].strip().lower() == user_ans.strip().lower():
                                is_correct = True
                        except Exception:
                            pass
                    score = 1 if is_correct else 0
                    total_score += score
                    fb = {
                        "id": q.id,
                        "type": q.type,
                        "question": q.prompt,
                        "your_answer": user_ans,
                        "correct_answer": q.answer,
                        "is_correct": is_correct,
                        "explanation": q.explanation
                    }
                    feedback_list.append(fb)
                else:
                    max_score += 100
                    if google_api_key:
                        grade = grade_open_answer_with_google(google_api_key, q.prompt, q.answer, user_ans, difficulty)
                    else:
                        grade = simple_fallback_grade(q.answer, user_ans)
                    score = int(grade.get("score", 0))
                    total_score += score
                    fb = {
                        "id": q.id,
                        "type": q.type,
                        "question": q.prompt,
                        "your_answer": user_ans,
                        "score": score,
                        "feedback": grade.get("feedback", ""),
                        "explanation": q.explanation
                    }
                    feedback_list.append(fb)
        overall_percent = int(100 * total_score / max_score) if max_score > 0 else 0
        st.session_state["feedback"] = {"overall_percent": overall_percent, "details": feedback_list}
        st.success(f"Grading complete — Score: {overall_percent}%")

# Show feedback
fb = st.session_state.get("feedback")
if fb:
    st.header("Detailed Feedback")
    st.write(f"Overall score: {fb['overall_percent']}%")
    for item in fb["details"]:
        st.markdown(f"### Question {item['id']}")
        st.write(item["question"])
        if item["type"] == "multiple_choice":
            st.write(f"- Your answer: **{item['your_answer']}**")
            correctness = "Correct ✅" if item.get("is_correct") else "Incorrect ❌"
            st.write(f"- Result: **{correctness}**")
            st.write(f"- Correct answer: **{item.get('correct_answer', '—')}**")
            st.write(f"- Explanation: {item.get('explanation', '')}")
        else:
            st.write(f"- Your answer (submitted):")
            st.write(item.get("your_answer", ""))
            st.write(f"- Score: **{item.get('score', 0)} / 100**")
            st.write(f"- Feedback: {item.get('feedback', '')}")
            st.write(f"- Reference explanation: {item.get('explanation', '')}")
    st.markdown("---")
    st.button("Regenerate quiz or try again", on_click=lambda: st.session_state.update({"quiz": None, "answers": {}, "feedback": None}))

st.markdown("----")
st.write("Notes: Paste your Google API key into the sidebar field. Use the 'Test API Key' button to run a minimal check (diagnostics appear in the sidebar). If Google generation fails the app falls back to a built-in generator so you can continue.")
