"""
Streamlit Quiz App — fixed Google model (Gemini 2.5 Flash Lite)

This version uses the fixed model "gemini-2.5-flash-lite" and improves diagnostics and parsing
to avoid silent failures that returned None/404/400 before.

API key is NOT requested in the UI. The app looks for the key in this order:
  1) st.secrets["GOOGLE_API_KEY"]  (recommended for Streamlit Cloud / local .streamlit/secrets.toml)
  2) environment variable GOOGLE_API_KEY

If no key is found or the call fails, the app falls back to an internal self-contained generator/grader.

Notes:
- I added robust endpoint handling (tries /v1 then /v1beta2), detailed sidebar debug output
  when Google responses fail, and more tolerant JSON extraction.
- Keep your key in st.secrets or as an environment variable. Do not paste it into the UI.
"""
from dataclasses import dataclass
import json
import random
import re
import os
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

# -----------------------
# Configuration (fixed model)
# -----------------------
GOOGLE_MODEL = "gemini-2.5-flash-lite"  # fixed model, no UI option to change

# -----------------------
# Data structures
# -----------------------
@dataclass
class QuizQuestion:
    id: int
    type: str  # "multiple_choice" or "open"
    prompt: str
    choices: Optional[List[str]]  # only for multiple choice
    answer: str  # canonical/expected answer (for MC this is the correct choice text or index)
    explanation: str  # detailed explanation / answer key

# -----------------------
# Helper: JSON extraction
# -----------------------
def extract_json(text: str) -> Optional[str]:
    """Attempt to find the first valid JSON object/array in text."""
    if not text:
        return None
    candidates = []
    # {...} spans
    brace_spans = []
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
                brace_spans.append((start, i + 1))
                start = None
    for s, e in brace_spans:
        candidates.append(text[s:e])
    # [...] spans
    arr_spans = []
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
                arr_spans.append((start, i + 1))
                start = None
    for s, e in arr_spans:
        candidates.append(text[s:e])
    # try to parse candidates
    for c in candidates:
        try:
            json.loads(c)
            return c
        except Exception:
            continue
    # try whole text as last resort
    try:
        json.loads(text)
        return text
    except Exception:
        return None

# -----------------------
# Built-in fallback generator/grader
# -----------------------
def simple_fallback_quiz(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> List[QuizQuestion]:
    """
    Simple internal quiz generator producing basic MC or open questions.
    """
    source = (context_text or subject or "General knowledge").strip()
    sentences = re.split(r'(?<=[.!?])\s+', source)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    questions: List[QuizQuestion] = []
    for i in range(n_questions):
        if i < len(sentences):
            sent = sentences[i]
        else:
            sent = (subject or "This subject") + " overview."
        q_text = sent.rstrip('.')
        q_prompt = f"What is an important fact or idea in this statement: {q_text}?"
        if q_type == "multiple_choice":
            words = re.findall(r"\w+", sent)
            key = words[-1] if words else "concept"
            choices = [key, key + "s", "another concept", "not related"]
            random.shuffle(choices)
            correct = choices[0]
            explanation = f"The important term is '{key}' based on the sentence: \"{sent}\"."
            questions.append(QuizQuestion(id=i+1, type="multiple_choice", prompt=q_prompt, choices=choices, answer=correct, explanation=explanation))
        else:
            explanation = f"A good answer should refer to: {sent}"
            questions.append(QuizQuestion(id=i+1, type="open", prompt=q_prompt, choices=None, answer=sent, explanation=explanation))
    return questions

def simple_fallback_grade(reference: str, student_answer: str) -> Dict[str, Any]:
    ref_words = set(re.findall(r"\w+", reference.lower()))
    stud_words = set(re.findall(r"\w+", student_answer.lower()))
    common = ref_words & stud_words
    if not ref_words:
        score = 0
    else:
        score = int(100 * len(common) / len(ref_words))
    feedback = f"Fallback grading: {len(common)} overlapping key words. Expected key points: {reference}"
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

Output MUST be valid JSON. Output a top-level object like: {{
  "questions": [ ... ]
}}

Each question object must contain:
- "id": integer
- "type": either "multiple_choice" or "open"
- "prompt": the question text
- "choices": an array of strings (for multiple_choice only; for open type set to null or omit)
- "answer": for multiple_choice: the exact text of the correct choice; for open: a short model answer or keypoint summary
- "explanation": a detailed explanation and model answer to be shown after the user answers

If q_type == "mix", then include a mix of types (roughly half MC and half open).
Be concise but ensure clarity. DO NOT output anything except valid JSON (no commentary, no labels)."""
    return prompt

def build_grading_prompt(question: str, reference: str, student_answer: str, difficulty: str) -> str:
    return f"""You are an objective grader. Grade the student's answer from 0 to 100 based on correctness and completeness relative to the reference answer.
Question: {question}
Reference / expected answer: {reference}
Student answer: {student_answer}
Difficulty: {difficulty}

Return ONLY valid JSON like: {{ "score": <int 0-100>, "feedback": "<detailed feedback explaining strengths, weaknesses, and how to improve>" }}"""

# -----------------------
# Google Generative API interaction (robust and diagnostic)
# -----------------------
def call_google_generate(model: str, api_key: str, prompt_text: str, max_output_tokens: int = 1024, temperature: float = 0.2, timeout: int = 60) -> Optional[str]:
    """
    Call Google Generative Language API (tries v1 then v1beta2 endpoints).
    Returns generated text or None on failure. Writes diagnostics to the sidebar to help debug.
    """
    if not api_key:
        return None

    # prepare model path
    if model.startswith("models/"):
        model_path = model
    else:
        model_path = f"models/{model}"

    # Try both v1 and v1beta2 endpoints (some projects have different availabilities)
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/{model_path}:generateText?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta2/{model_path}:generateText?key={api_key}"
    ]

    last_status = None
    last_body = None

    for url in endpoints:
        try:
            headers = {"Content-Type": "application/json"}
            body = {"prompt": {"text": prompt_text}, "temperature": temperature, "maxOutputTokens": max_output_tokens}
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            status = resp.status_code
            text = resp.text
            # Attach diagnostics to sidebar (helpful during debug)
            try:
                st.sidebar.text(f"Google endpoint tried: {url}")
                st.sidebar.text(f"Status: {status}")
                # show a slice of the body to avoid huge UI blocks, but include full in logs
                st.sidebar.text_area("Google raw response (truncated)", value=(text[:4000] + ("...[truncated]" if len(text) > 4000 else "")), height=220)
            except Exception:
                # fallback to console prints if sidebar not available
                print("Google endpoint tried:", url)
                print("Status:", status)
                print("Body:", text[:2000])

            if not resp.ok:
                last_status = status
                last_body = text
                # try next endpoint
                continue

            # parse JSON and extract text robustly
            data = resp.json()
            # Common formats: {"candidates":[{"content":"..."}]} or {"candidates":[{"output":"..."}]} etc.
            if isinstance(data, dict):
                # early error detection
                if data.get("error"):
                    last_status = status
                    last_body = json.dumps(data)
                    continue
                candidates = data.get("candidates") or data.get("responses") or []
                if isinstance(candidates, list) and len(candidates) > 0:
                    first = candidates[0]
                    # support multiple possible keys
                    for k in ("content", "output", "text", "generated_text"):
                        if isinstance(first, dict) and first.get(k):
                            return first.get(k)
                    # if first is a string
                    if isinstance(first, str) and first.strip():
                        return first
                # Some responses may use "response" or other shapes; try to stringify helpful parts
                # try to find any string in the JSON recursively
                def find_first_string(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            s = find_first_string(v)
                            if s:
                                return s
                    if isinstance(obj, list):
                        for el in obj:
                            s = find_first_string(el)
                            if s:
                                return s
                    return None
                found = find_first_string(data)
                if found:
                    return found
            # fallback: return the raw text so caller can inspect
            return text
        except Exception as e:
            last_status = "exception"
            last_body = str(e)
            # continue to next endpoint
            continue

    # If we reach here, both endpoints failed
    try:
        st.sidebar.error(f"Google calls failed. Last status: {last_status}")
        st.sidebar.text_area("Last Google response body (for debugging)", value=(last_body or "no body"), height=220)
    except Exception:
        print("Google calls failed. Last status:", last_status)
        print("Last body:", last_body)
    return None

def generate_quiz_with_google(api_key: str, context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> Optional[List[QuizQuestion]]:
    """
    Generate quiz using Google Generative API with a fixed model.
    """
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
            qid = int(q.get("id", len(qs) + 1))
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
        # if parsing fails, write raw output into sidebar for inspection and return None
        try:
            st.sidebar.text_area("Failed to parse JSON from model output (raw):", value=text, height=300)
        except Exception:
            print("Failed to parse JSON from model output. Raw text:", text)
        return None

def grade_open_answer_with_google(api_key: str, question: str, reference: str, student_answer: str, difficulty: str) -> Dict[str, Any]:
    """
    Use Google model (fixed) to grade an open answer. Returns a dict {"score": int, "feedback": str}.
    Falls back to simple overlap scoring on failure.
    """
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
        # fallback naive
        return simple_fallback_grade(reference, student_answer)

# -----------------------
# Streamlit UI (no API key input in UI)
# -----------------------
st.set_page_config(page_title="Quiz Generator (Gemini 2.5 Flash Lite)", layout="wide")
st.title("Quiz Generator — fixed Google model: gemini-2.5-flash-lite — API key not requested in UI")

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

st.sidebar.markdown("---")
st.sidebar.markdown("This app will try to use Google Generative API with model 'gemini-2.5-flash-lite' if a key is configured in st.secrets or the GOOGLE_API_KEY environment variable. The key is NOT requested or shown in the UI. Diagnostics from Google calls (status and raw response) will appear in the sidebar to help debug failures.")

# retrieve API key securely (st.secrets -> env)
google_api_key = None
try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    google_api_key = None
if not google_api_key:
    google_api_key = os.environ.get("GOOGLE_API_KEY")

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
            # Try Google if API key present
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
    answers = {}
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
st.write("Notes: The app attempts to call Google's Generative API with model 'gemini-2.5-flash-lite'. Diagnostics (endpoint tried, HTTP status, and a truncated raw response) will appear in the sidebar to help you see why previous calls returned 400/404. If no valid response is obtained, the internal fallback generator is used.")
