"""
QuizGen AI — Streamlit (Python) port of the provided React/TS project.

Features:
- Setup screen: paste Gemini / Google Generative API key (optional), choose mode (subject/context),
  enter input, count, difficulty, and question type (multiple choice / open / mixed).
- Generate quiz using Google Generative API (gemini-2.5-flash-lite preferred). If API key missing
  or the cloud call fails, a local fallback generator is used.
- Run through quiz (navigation, answer storage).
- Submit and grade: grading via Google model when API key provided, otherwise fallback heuristic scoring.
- Simple diagnostics shown in the sidebar when cloud calls fail.

Run:
  pip install -r requirements.txt
  streamlit run app.py

Notes:
- This is a pragmatic single-file conversion from the React app you provided.
- The app accepts a key in the UI (sidebar) for convenience. If you prefer to hide it, you can remove
  the sidebar input and set GOOGLE_API_KEY via environment or Streamlit secrets.
"""
from dataclasses import dataclass
import json
import re
import random
import os
from typing import List, Optional, Dict, Any
import requests
import streamlit as st

# -----------------------
# Data models
# -----------------------
@dataclass
class Question:
    id: int
    text: str
    type: str  # 'multiple_choice' | 'open_ended'
    options: Optional[List[str]] = None

@dataclass
class GradingResult:
    questionId: int
    score: int  # 0-100
    feedback: str
    correctAnswer: str

# -----------------------
# Configuration
# -----------------------
PREFERRED_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "text-bison-001"]
TRY_ENDPOINTS = [
    "https://generativelanguage.googleapis.com/v1/{model_path}:generateText?key={key}",
    "https://generativelanguage.googleapis.com/v1beta2/{model_path}:generateText?key={key}"
]

# -----------------------
# Utilities
# -----------------------
def extract_json(text: str) -> Optional[str]:
    """Attempt to extract first JSON object/array in text."""
    if not text:
        return None
    # look for {...} or [...]
    candidates = []
    # braces
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            stack.append('{')
        elif ch == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:i+1])
                start = None
    # arrays
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '[':
            if start is None:
                start = i
            stack.append('[')
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
    # try whole thing
    try:
        json.loads(text)
        return text
    except Exception:
        return None

def simple_fallback_quiz(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> List[Question]:
    source = (context_text or subject or "General knowledge").strip()
    sentences = re.split(r'(?<=[.!?])\s+', source)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    qs = []
    for i in range(n_questions):
        sent = sentences[i] if i < len(sentences) else (subject or "This subject") + " overview."
        q_text = sent.rstrip('.')
        if q_type == "multiple_choice":
            words = re.findall(r"\w+", sent)
            key = words[-1] if words else "concept"
            choices = [key, key + "s", "another concept", "not related"]
            random.shuffle(choices)
            qs.append(Question(id=i+1, text=f"What is an important fact in this statement: {q_text}?", type="multiple_choice", options=choices))
        elif q_type == "open_ended":
            qs.append(Question(id=i+1, text=f"What is an important fact in this statement: {q_text}?", type="open_ended", options=None))
        else:  # mixed
            if i % 2 == 0:
                words = re.findall(r"\w+", sent)
                key = words[-1] if words else "concept"
                choices = [key, key + "s", "another concept", "not related"]
                random.shuffle(choices)
                qs.append(Question(id=i+1, text=f"What is an important fact in this statement: {q_text}?", type="multiple_choice", options=choices))
            else:
                qs.append(Question(id=i+1, text=f"What is an important fact in this statement: {q_text}?", type="open_ended", options=None))
    return qs

def simple_fallback_grade(questions: List[Question], answers: Dict[int, str]) -> List[GradingResult]:
    results = []
    for q in questions:
        ref = q.text
        stud = answers.get(q.id, "").strip().lower()
        if q.type == "multiple_choice" and q.options:
            correct = q.options[0]  # fallback: first option
            score = 100 if stud and stud == correct else 0
            feedback = "Correct." if score == 100 else f"Incorrect. Expected: {correct}"
            results.append(GradingResult(questionId=q.id, score=score, feedback=feedback, correctAnswer=correct))
        else:
            ref_words = set(re.findall(r"\w+", ref.lower()))
            stud_words = set(re.findall(r"\w+", stud.lower()))
            common = ref_words & stud_words
            score = int(100 * len(common) / len(ref_words)) if ref_words else 0
            feedback = f"Fallback grading: {len(common)} overlapping key words."
            results.append(GradingResult(questionId=q.id, score=score, feedback=feedback, correctAnswer=ref))
    return results

# -----------------------
# Google Generative API interaction (models fallback & diagnostics)
# -----------------------
def call_google_generate(model: str, api_key: str, prompt_text: str, max_output_tokens: int = 1024, temperature: float = 0.0, timeout: int = 60) -> Optional[str]:
    """
    Try a model using v1 and v1beta2 endpoints. Return string output or None.
    Adds diagnostics to sidebar if available.
    """
    if not api_key:
        return None
    model_path = model if model.startswith("models/") else f"models/{model}"
    headers = {"Content-Type": "application/json"}
    body = {"prompt": {"text": prompt_text}, "temperature": temperature, "maxOutputTokens": max_output_tokens}
    last_status = None
    last_body = None
    for endpoint_template in TRY_ENDPOINTS:
        url = endpoint_template.format(model_path=model_path, key=api_key)
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            status = resp.status_code
            text = resp.text
            last_status = status
            last_body = text
            # Diagnostics
            try:
                st.sidebar.markdown(f"**Google endpoint tried:** `{url}`")
                st.sidebar.markdown(f"**HTTP status:** `{status}`")
                st.sidebar.text_area("Raw Google response (truncated)", value=(text[:4000] + ("...[truncated]" if len(text) > 4000 else "")), height=220)
            except Exception:
                # ignore sidebar failures
                pass
            if not resp.ok:
                continue
            data = resp.json()
            if isinstance(data, dict):
                if data.get("error"):
                    last_body = json.dumps(data)
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
            # fallback: find any string in JSON
            def find_string(obj):
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        s = find_string(v)
                        if s:
                            return s
                if isinstance(obj, list):
                    for e in obj:
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
            last_body = str(e)
            try:
                st.sidebar.error(f"Google call exception: {e}")
            except Exception:
                pass
            continue
    # All endpoints attempted
    try:
        st.sidebar.error(f"All Google endpoints failed. Last status: {last_status}")
        st.sidebar.text_area("Last Google response body (debug)", value=(last_body or "no body"), height=220)
    except Exception:
        pass
    return None

def generate_quiz_with_google(api_key: str, context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> Optional[List[Question]]:
    if not api_key:
        return None
    prompt = build_generation_prompt(context_text, subject, difficulty, n_questions, q_type)
    # try preferred models in order; on 404 try next model
    last_text = None
    for model in PREFERRED_MODELS:
        text = call_google_generate(model, api_key, prompt, max_output_tokens=1024, temperature=0.0)
        if not text:
            continue
        last_text = text
        json_str = extract_json(text) or text
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                questions = []
                for q in parsed:
                    qid = int(q.get("id", len(questions) + 1))
                    qtype = q.get("type", "open_ended")
                    qtext = q.get("text") or q.get("prompt") or ""
                    options = q.get("options") or q.get("choices") or None
                    questions.append(Question(id=qid, text=qtext, type=qtype, options=options))
                return questions[:n_questions]
            # support {"questions": [...]}
            if isinstance(parsed, dict) and parsed.get("questions"):
                parsed_list = parsed.get("questions")
                questions = []
                for q in parsed_list:
                    qid = int(q.get("id", len(questions) + 1))
                    qtype = q.get("type", "open_ended")
                    qtext = q.get("text") or q.get("prompt") or ""
                    options = q.get("options") or q.get("choices") or None
                    questions.append(Question(id=qid, text=qtext, type=qtype, options=options))
                return questions[:n_questions]
        except Exception:
            # if parse fails, continue to next model
            try:
                st.sidebar.text_area("Failed to parse JSON from model output (raw):", value=text, height=300)
            except Exception:
                pass
            continue
    return None

def grade_with_google(api_key: str, questions: List[Question], answers: Dict[int,str], original_context: str) -> Optional[List[GradingResult]]:
    if not api_key:
        return None
    payload = []
    for q in questions:
        payload.append({
            "questionId": q.id,
            "questionText": q.text,
            "userAnswer": answers.get(q.id, "(No Answer)"),
            "type": q.type
        })
    prompt = build_grading_prompt(payload, original_context)
    for model in PREFERRED_MODELS:
        text = call_google_generate(model, api_key, prompt, max_output_tokens=512, temperature=0.0)
        if not text:
            continue
        json_str = extract_json(text) or text
        try:
            parsed = json.loads(json_str)
            results = []
            if isinstance(parsed, list):
                for r in parsed:
                    qid = int(r.get("questionId"))
                    score = int(r.get("score", 0))
                    feedback = r.get("feedback","")
                    correct = r.get("correctAnswer") or r.get("correct_answer") or ""
                    results.append(GradingResult(questionId=qid, score=score, feedback=feedback, correctAnswer=correct))
                return results
            # tolerate dict-shaped responses
            if isinstance(parsed, dict) and parsed.get("results"):
                for r in parsed.get("results"):
                    qid = int(r.get("questionId"))
                    score = int(r.get("score", 0))
                    feedback = r.get("feedback","")
                    correct = r.get("correctAnswer") or r.get("correct_answer") or ""
                    results.append(GradingResult(questionId=qid, score=score, feedback=feedback, correctAnswer=correct))
                return results
        except Exception:
            try:
                st.sidebar.text_area("Failed to parse grading JSON (raw):", value=text, height=300)
            except Exception:
                pass
            continue
    return None

# -----------------------
# Prompt builders
# -----------------------
def build_generation_prompt(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> str:
    source = context_text if context_text else f"Subject: {subject}"
    type_instruction = ""
    if q_type == "multiple_choice":
        type_instruction = "All questions must be multiple choice with 4 options."
    elif q_type == "open_ended":
        type_instruction = "All questions must be open-ended textual questions."
    else:
        type_instruction = "Mix multiple choice and open-ended questions evenly."

    prompt = f"""
You are a helpful quiz writer. Create a quiz of {n_questions} question(s) in JSON format ONLY.
Context:
\"\"\"{source}\"\"\"

Difficulty: {difficulty}
Question type preference: {q_type}

IMPORTANT: Output ONLY valid JSON and nothing else.
Return either a JSON array of question objects, or an object with a top-level "questions" array.

Each question object must contain:
- "id": integer
- "type": "multiple_choice" or "open_ended"
- "text": the question text
- "options": array of strings (for multiple_choice) or empty/null for open_ended

Do not add any explanation fields in generation output. Keep it strictly JSON.
{type_instruction}
"""
    return prompt

def build_grading_prompt(payload: List[Dict[str, Any]], original_context: str) -> str:
    return f"""
You are a strict but fair grader. Grade each answer from 0 to 100 and provide concise feedback and the correct answer.

Context:
\"\"\"{original_context}\"\"\"

Quiz Data:
{json.dumps(payload, indent=2)}

Return a JSON array where each element is an object with:
- "questionId": integer
- "score": integer 0-100
- "feedback": string
- "correctAnswer": string

Output only valid JSON.
"""

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="QuizGen AI (Python)", layout="wide")
st.title("QuizGen AI — Python / Streamlit port")

# Sidebar: API key + basic diagnostics
st.sidebar.header("Google API / Gemini Key")
st.sidebar.info("Paste a Google Generative API key (optional). If provided the app will attempt to use Gemini / text-bison models for generation and grading. If omitted, a local fallback will be used.")
ui_api_key = st.sidebar.text_input("Google API Key (paste here)", type="password")
if ui_api_key:
    st.sidebar.success("API key provided (not shown).")

# Setup form
st.sidebar.markdown("---")
st.sidebar.header("Quiz Setup")
mode = st.sidebar.radio("Source mode", ("subject", "context"))
if mode == "subject":
    subject = st.sidebar.text_input("Subject (short)", value="Photosynthesis")
    context_text = ""
else:
    context_text = st.sidebar.text_area("Paste context text", height=220)
    subject = ""

question_count = st.sidebar.slider("Number of questions", min_value=1, max_value=20, value=5)
difficulty = st.sidebar.selectbox("Difficulty", ("easy", "medium", "hard"))
q_type = st.sidebar.selectbox("Question type", ("mixed", "multiple_choice", "open_ended"))
randomize = st.sidebar.checkbox("Randomize question order", value=True)
test_key = st.sidebar.button("Test API Key")

# persist state
if "state" not in st.session_state:
    st.session_state.state = {
        "questions": [],
        "answers": {},
        "results": None,
        "current": 0,
        "is_generating": False,
        "is_grading": False,
        "error": None,
        "api_ok": False
    }

state = st.session_state.state

# Test API key button action
if test_key:
    if not ui_api_key:
        st.sidebar.error("Please paste a key into the field first.")
    else:
        st.sidebar.info("Testing key using text-bison-001 (or trying Gemini) ...")
        test_prompt = 'Respond with only the JSON: {"ok": true}'
        out = call_google_generate(PREFERRED_MODELS[0], ui_api_key, test_prompt, max_output_tokens=64, temperature=0.0)
        if not out:
            # try text-bison as explicit test if preferred model failed
            out2 = call_google_generate("text-bison-001", ui_api_key, test_prompt, max_output_tokens=64, temperature=0.0)
            if out2:
                st.sidebar.success("Key appears to work with text-bison-001 (fallback).")
                state["api_ok"] = True
            else:
                st.sidebar.error("No usable response from Google Generative API. Check key, API enablement, and billing.")
                state["api_ok"] = False
        else:
            st.sidebar.success("API key produced a response (inspect sidebar diagnostics above).")
            state["api_ok"] = True

# Generate quiz action
if st.sidebar.button("Generate Quiz"):
    state["questions"] = []
    state["answers"] = {}
    state["results"] = None
    state["current"] = 0
    state["error"] = None
    state["is_generating"] = True

    # call cloud if key provided, else fallback
    try:
        qs = None
        if ui_api_key:
            qs = generate_quiz_with_google(ui_api_key, context_text, subject, difficulty, question_count, q_type)
            if qs is None:
                st.warning("Cloud generation failed or returned invalid output; falling back to local generator.")
        if qs is None:
            qs = simple_fallback_quiz(context_text, subject, difficulty, question_count, q_type)
        if randomize:
            random.shuffle(qs)
        state["questions"] = qs
        state["is_generating"] = False
    except Exception as e:
        state["error"] = str(e)
        state["is_generating"] = False
        st.error(f"Failed to generate quiz: {e}")

# Main rendering logic
def show_setup():
    st.markdown("### Get started")
    st.write("Use the sidebar to paste an API key (optional), provide a subject or context, and configure the quiz. Click Generate Quiz to begin.")

def show_quiz_runner():
    questions: List[Question] = state["questions"]
    if not questions:
        show_setup()
        return
    idx = state["current"]
    q = questions[idx]
    st.header(f"Question {idx+1} of {len(questions)}")
    st.subheader(q.text)
    if q.type == "multiple_choice" and q.options:
        # show options as buttons
        for opt in q.options:
            sel = st.radio("Options", q.options, index=q.options.index(state["answers"].get(q.id, q.options[0])) if state["answers"].get(q.id) in (q.options or []) else 0, key=f"radio_{q.id}")
        # store selected
        state["answers"][q.id] = st.session_state.get(f"radio_{q.id}", state["answers"].get(q.id, ""))
    else:
        txt = st.text_area("Your answer", value=state["answers"].get(q.id, ""), key=f"answer_{q.id}", height=160)
        state["answers"][q.id] = st.session_state.get(f"answer_{q.id}", txt)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Previous", disabled=(idx==0)):
            state["current"] = max(0, idx-1)
    with col2:
        if st.button("Next", disabled=(idx==len(questions)-1)):
            state["current"] = min(len(questions)-1, idx+1)
    with col3:
        if st.button("Submit & Grade", disabled=False if len(questions)>0 else True):
            # grade
            state["is_grading"] = True
            st.experimental_rerun()

def show_results():
    results: List[GradingResult] = state["results"] or []
    questions: List[Question] = state["questions"]
    if not results:
        st.write("No results to show.")
        return
    total = round(sum(r.score for r in results) / len(results))
    st.markdown(f"## Final Score: **{total}%**")
    for r in results:
        q = next((x for x in questions if x.id == r.questionId), None)
        if not q:
            continue
        st.markdown(f"### Q{r.questionId}: {q.text}")
        st.write(f"- Score: **{r.score} / 100**")
        st.write(f"- AI feedback: {r.feedback}")
        if r.score < 100:
            st.write(f"- Correct answer: **{r.correctAnswer}**")
    if st.button("Create New Quiz"):
        state["questions"] = []
        state["answers"] = {}
        state["results"] = None
        state["current"] = 0
        state["error"] = None
        st.experimental_rerun()

# If grading requested
if state["is_grading"]:
    # grade now
    try:
        generated_results = None
        if ui_api_key:
            generated_results = grade_with_google(ui_api_key, state["questions"], state["answers"], context_text or subject)
            if generated_results is None:
                st.warning("Cloud grading failed or returned invalid output; using fallback grading.")
        if generated_results is None:
            generated_results = simple_fallback_grade(state["questions"], state["answers"])
        state["results"] = generated_results
        state["is_grading"] = False
        st.experimental_rerun()
    except Exception as e:
        state["error"] = str(e)
        state["is_grading"] = False
        st.error(f"Failed to grade: {e}")

# Render screens
if state["results"]:
    show_results()
else:
    if not state["questions"]:
        show_setup()
    else:
        show_quiz_runner()

# show errors if any
if state.get("error"):
    st.error(state["error"])
