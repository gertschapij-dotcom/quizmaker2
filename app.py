"""
Streamlit Quiz App using Google Generative AI (preferred if you supply an API key) or a built-in fallback.

How it works:
- Accepts either an uploaded context text file OR a short subject line.
- Lets user choose difficulty, number of questions, and question type (multiple choice, open answer, mix).
- If you provide a Google Generative API key in the sidebar, the app will call the Google Generative API (e.g. text-bison-001) to generate quizzes and grade open answers.
- If no API key is provided (or calls fail), the app uses a simple self-contained fallback generator & grader.
- Everything lives inside this single file; to run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from dataclasses import dataclass
import json
import random
import re
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

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
# JSON extraction helper
# -----------------------
def extract_json(text: str) -> Optional[str]:
    """
    Attempt to extract a JSON object/array from text.
    Returns the JSON substring if found, else None.
    """
    if not text:
        return None
    candidates = []
    # find {...} spans
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
    # find [...] spans
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

    for c in candidates:
        try:
            json.loads(c)
            return c
        except Exception:
            continue
    # as a last resort, try to parse the whole thing
    try:
        json.loads(text)
        return text
    except Exception:
        return None

# -----------------------
# Fallback generator
# -----------------------
def simple_fallback_quiz(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> List[QuizQuestion]:
    """
    A very simple internal quiz generator (no external calls). Produces basic MC or open questions.
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
Be concise but ensure clarity. Do not include any commentary outside the JSON."""
    return prompt

def build_grading_prompt(question: str, reference: str, student_answer: str, difficulty: str) -> str:
    return f"""You are an objective grader. Grade the student's answer from 0 to 100 based on correctness and completeness relative to the reference answer.
Question: {question}
Reference / expected answer: {reference}
Student answer: {student_answer}
Difficulty: {difficulty}

Return ONLY valid JSON like: {{ "score": <int 0-100>, "feedback": "<detailed feedback explaining strengths, weaknesses, and how to improve>" }}"""

# -----------------------
# Google Generative API calls
# -----------------------
def call_google_generate(model: str, api_key: str, prompt_text: str, max_output_tokens: int = 1024, temperature: float = 0.2, timeout: int = 60) -> Optional[str]:
    """
    Call Google Generative Language API (REST) to generate text.
    model e.g. "models/text-bison-001"
    api_key: your Google API key (kept secret in the sidebar)
    Returns generated text or None on failure.
    """
    if not api_key:
        return None
    # Endpoint pattern: https://generativelanguage.googleapis.com/v1beta2/{model}:generateText?key=API_KEY
    # Accept model param as either "text-bison-001" or "models/text-bison-001"
    if not model:
        model = "text-bison-001"
    if not model.startswith("models/"):
        model_path = f"models/{model}"
    else:
        model_path = model
    url = f"https://generativelanguage.googleapis.com/v1beta2/{model_path}:generateText?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "prompt": {"text": prompt_text},
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # expected: data["candidates"][0]["content"]
        if isinstance(data, dict) and "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
            content = data["candidates"][0].get("content")
            if content is None:
                # sometimes "output" or "content"? try keys
                for k in ("output", "content", "text"):
                    if k in data["candidates"][0]:
                        return data["candidates"][0].get(k)
            return content
        # fallback try stringifying
        return json.dumps(data)
    except Exception:
        return None

def generate_quiz_with_google(model: str, api_key: str, context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> Optional[List[QuizQuestion]]:
    """
    Generate quiz using Google Generative API.
    """
    if not api_key:
        return None
    prompt = build_generation_prompt(context_text, subject, difficulty, n_questions, q_type)
    text = call_google_generate(model, api_key, prompt, max_output_tokens=1024, temperature=0.2)
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
        return None

def grade_open_answer_with_google(model: str, api_key: str, question: str, reference: str, student_answer: str, difficulty: str) -> Dict[str, Any]:
    """
    Use Google model to grade an open answer. Returns a dict {"score": int, "feedback": str}.
    Falls back to simple overlap scoring on failure.
    """
    if not api_key:
        # fallback to naive scoring
        ref_words = set(re.findall(r"\w+", reference.lower()))
        stud_words = set(re.findall(r"\w+", student_answer.lower()))
        common = ref_words & stud_words
        if not ref_words:
            score = 0
        else:
            score = int(100 * len(common) / len(ref_words))
        feedback = f"Fallback grading: {len(common)} overlapping key words. Expected key points: {reference}"
        return {"score": score, "feedback": feedback}
    prompt = build_grading_prompt(question, reference, student_answer, difficulty)
    text = call_google_generate(model, api_key, prompt, max_output_tokens=400, temperature=0.0)
    if not text:
        # fallback
        return {"score": 0, "feedback": "Grading failed (no response)."}
    json_str = extract_json(text) or text
    try:
        parsed = json.loads(json_str)
        score = int(parsed.get("score", 0))
        feedback = parsed.get("feedback", "")
        return {"score": score, "feedback": feedback}
    except Exception:
        # fallback naive
        ref_words = set(re.findall(r"\w+", reference.lower()))
        stud_words = set(re.findall(r"\w+", student_answer.lower()))
        common = ref_words & stud_words
        score = int(100 * len(common) / len(ref_words)) if ref_words else 0
        feedback = f"Fallback grading (post-parse): {len(common)} overlapping key words. Expected key points: {reference}"
        return {"score": score, "feedback": feedback}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Google Generative Quiz Generator", layout="wide")
st.title("Quiz Generator — Google Generative API (optional) + Built-in Fallback")

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

st.sidebar.header("Google Generative API (optional)")
st.sidebar.write("Provide your Google API key to enable the Generative API (text-bison-001 or other model). If left blank the app will use the fully self-contained fallback generator.")
google_api_key = st.sidebar.text_input("Google API Key (optional)", type="password")
google_model = st.sidebar.text_input("Google model (e.g. text-bison-001)", value="text-bison-001")
st.sidebar.markdown("Google Generative API docs: https://developers.generativeai.google/")

st.sidebar.markdown("---")
st.sidebar.markdown("Privacy: your API key is used only by this running app. Do not paste keys you don't trust. If you don't provide a key, the app uses the internal fallback generator.")

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
            # Try Google if API key provided
            if google_api_key:
                qs = generate_quiz_with_google(google_model, google_api_key, context_text, subject, difficulty, n_questions, q_type)
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
                        grade = grade_open_answer_with_google(google_model, google_api_key, q.prompt, q.answer, user_ans, difficulty)
                    else:
                        grade = grade_open_answer_with_google("", "", q.prompt, q.answer, user_ans, difficulty)
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
st.write("Tips: For best results, paste a Google Generative API key in the sidebar and keep your model as 'text-bison-001' (or another available Generative model). If no key is provided, the app uses a simple built-in generator.")
