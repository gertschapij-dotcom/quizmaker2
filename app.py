"""
Streamlit Quiz App using a free self-contained local LLM (GPT4All recommended)

How it works (high level):
- Accepts either an uploaded context text file OR a short subject line.
- Lets user choose difficulty, number of questions, and question type (multiple choice, open answer, mix).
- Uses a local LLM (gpt4all via the gpt4all Python package) to generate a quiz in structured JSON.
- Presents the quiz, collects user answers, and then provides detailed feedback:
  - For multiple-choice: immediate correctness and explanation from the generated key.
  - For open-ended: uses the LLM to grade / score the open answer and produce detailed feedback.
- Includes a small built-in fallback generator if the local model isn't available.

Notes:
- To use a local LLM, install the 'gpt4all' Python package and download a compatible model (see README).
- The app attempts to be robust to model output; it will try to parse JSON out of the LLM response.
"""

import streamlit as st
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

# Try to import gpt4all; if not available we'll use fallback
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except Exception:
    GPT4ALL_AVAILABLE = False

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
# Utility functions
# -----------------------
def extract_json(text: str) -> Optional[str]:
    """
    Attempt to extract a JSON object/array from text.
    Returns the JSON substring if found, else None.
    """
    # Try to find the first {...} or [...] block that parses
    # We'll search for top-level braces and attempt loads
    candidates = []
    # find all {...} spans
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
    # also try arrays
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
    return None

def simple_fallback_quiz(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> List[QuizQuestion]:
    """
    Very small heuristic quiz generator as fallback when no local LLM is available.
    It will create straightforward factual or conceptual questions by extracting sentences.
    """
    # Split text into sentences (naive)
    sentences = re.split(r'(?<=[.!?])\s+', (context_text or subject).strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    questions = []
    for i in range(n_questions):
        if i < len(sentences):
            sent = sentences[i]
        else:
            sent = (subject or "This subject") + " overview."
        q_prompt = f"Based on: {sent}\n\nWrite a short question asking about the key fact or idea in that sentence."
        # naive transform: take a noun phrase or convert statement to question
        # as fallback, reverse sentence to make a question
        q_text = sent.rstrip('.')
        q_text = "What is an important fact or idea in this statement: " + q_text + "?"
        # multiple choice fallback: generate choices by perturbing final words
        if q_type == "multiple_choice":
            choices = []
            words = re.findall(r"\w+", sent)
            if words:
                key = words[-1]
            else:
                key = "concept"
            choices = [key, key + "s", "another concept", "not related"]
            random.shuffle(choices)
            correct = choices[0]
            explanation = f"The important term is '{key}' based on the sentence."
            questions.append(QuizQuestion(id=i+1, type="multiple_choice", prompt=q_text, choices=choices, answer=correct, explanation=explanation))
        else:
            explanation = f"An appropriate answer should mention the content of: {sent}"
            questions.append(QuizQuestion(id=i+1, type="open", prompt=q_text, choices=None, answer=sent, explanation=explanation))
    return questions

# -----------------------
# LLM interaction
# -----------------------
def build_generation_prompt(context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str) -> str:
    """
    Build a clear prompt instructing the local model to output a JSON quiz.
    We ask for a JSON array of questions; each question must contain:
      - id (int)
      - type: "multiple_choice" or "open"
      - prompt (string)
      - choices (array of strings)  # for multiple_choice
      - answer (string)  # for MC: the correct choice text; for open: a short model answer/keypoints
      - explanation (string): detailed feedback / expected answer explanation
    We request only JSON in the output.
    """
    source = context_text if context_text else f"Subject: {subject}"
    prompt = f"""
You are a helpful quiz writer. Create a quiz of {n_questions} question(s) in JSON format ONLY.
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
Be concise but ensure clarity. Do not include any commentary outside the JSON.
"""
    return prompt

def generate_quiz_with_gpt4all(model_name: str, context_text: str, subject: str, difficulty: str, n_questions: int, q_type: str, timeout: int = 30) -> Optional[List[QuizQuestion]]:
    """
    Use GPT4All to generate a JSON quiz.
    model_name: model alias or filename for GPT4All
    Returns list of QuizQuestion or None on error.
    """
    if not GPT4ALL_AVAILABLE:
        return None
    try:
        # initialize model
        llm = GPT4All(model_name)
        prompt = build_generation_prompt(context_text, subject, difficulty, n_questions, q_type)
        # Show a short system prefacing to make the model deterministic-ish
        # Use streaming disabled for simplicity
        resp = llm.generate(prompt, max_tokens=1024, n_threads=4)
        text = resp.strip()
        json_str = extract_json(text) or text
        parsed = json.loads(json_str)
        qs = []
        for q in parsed.get("questions", []):
            qid = int(q.get("id", len(qs)+1))
            qtype = q.get("type", "open")
            prompt_text = q.get("prompt", "").strip()
            choices = q.get("choices", None)
            answer = q.get("answer", "")
            explanation = q.get("explanation", "")
            qs.append(QuizQuestion(id=qid, type=qtype, prompt=prompt_text, choices=choices, answer=answer, explanation=explanation))
        # Enforce requested count
        if len(qs) > n_questions:
            qs = qs[:n_questions]
        return qs
    except Exception as e:
        st.warning(f"LLM quiz generation failed: {e}")
        return None

def grade_open_answer_with_gpt4all(model_name: str, question: str, reference: str, student_answer: str, difficulty: str) -> Dict[str, Any]:
    """
    Use the local LLM to grade an open answer and return {score: int, feedback: str}
    """
    if not GPT4ALL_AVAILABLE:
        # fallback naive grading: compare overlap of words
        ref_words = set(re.findall(r"\w+", reference.lower()))
        stud_words = set(re.findall(r"\w+", student_answer.lower()))
        common = ref_words & stud_words
        if not ref_words:
            score = 0
        else:
            score = int(100 * len(common) / len(ref_words))
        feedback = f"Fallback grading: {len(common)} overlapping key words. Expected key points: {reference}"
        return {"score": score, "feedback": feedback}
    try:
        llm = GPT4All(model_name)
        grade_prompt = f"""
You are an objective grader. Grade the student's answer from 0 to 100 based on correctness and completeness relative to the reference answer.
Question: {question}
Reference / expected answer: {reference}
Student answer: {student_answer}
Difficulty: {difficulty}

Return ONLY valid JSON like: {{"score": <int 0-100>, "feedback": "<detailed feedback explaining strengths, weaknesses, and how to improve>"}}
"""
        resp = llm.generate(grade_prompt, max_tokens=512, n_threads=4)
        text = resp.strip()
        json_str = extract_json(text) or text
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        # fallback
        return {"score": 0, "feedback": f"Grading failed: {e}"}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Local LLM Quiz Generator", layout="wide")

st.title("Local LLM Quiz Generator — Streamlit")

st.sidebar.header("Quiz Source")
source_choice = st.sidebar.radio("Create quiz from:", ("Upload text file", "Subject / short description"))
uploaded_text = ""
subject_line = ""
if source_choice == "Upload text file":
    uploaded_file = st.sidebar.file_uploader("Upload a large text file (txt, md, pdf copy-paste). For PDFs, copy content into a .txt.", type=["txt", "md"])
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                try:
                    uploaded_text = raw.decode("utf-8")
                except Exception:
                    uploaded_text = raw.decode("latin-1")
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

st.sidebar.header("Local Model (optional)")
st.sidebar.write("If you have a local gpt4all / compatible model, specify its name or filename. If left blank the app will try to use a default gpt4all model if installed; otherwise it will use a simple fallback generator.")
model_name = st.sidebar.text_input("GPT4All model alias or filename", value="gpt4all-lora-quantized.bin" if GPT4ALL_AVAILABLE else "")

st.sidebar.markdown("---")
st.sidebar.markdown("Notes:\n- To use a local model install the gpt4all package and download a compatible model (see README).")

# Generate quiz
generate_button = st.sidebar.button("Generate Quiz")

# State to hold generated quiz
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
        st.sidebar.error("Please provide either an uploaded text file or a subject line.")
    else:
        with st.spinner("Generating quiz... This may take a little while if using a local LLM."):
            qs = None
            # Try LLM generation first if model present
            if model_name and GPT4ALL_AVAILABLE:
                qs = generate_quiz_with_gpt4all(model_name, context_text, subject, difficulty, n_questions, q_type)
            elif GPT4ALL_AVAILABLE and not model_name:
                # try default
                qs = generate_quiz_with_gpt4all("gpt4all-lora-quantized.bin", context_text, subject, difficulty, n_questions, q_type)
            if qs is None:
                # fallback
                st.info("Using fallback generator (no local model available or generation failed). The quiz will be simpler.")
                qs = simple_fallback_quiz(context_text, subject, difficulty, n_questions, q_type if q_type!="mix" else "mixed")
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
                # ensure choices exist
                choices = q.choices or ["A", "B", "C", "D"]
                selected = form.radio(f"Select answer for Q{q.id}", choices, key=f"q_{q.id}")
                answers[str(q.id)] = selected
            else:
                # open answer
                txt = form.text_area(f"Your answer for Q{q.id}", key=f"q_{q.id}", height=120)
                answers[str(q.id)] = txt
    submit = form.form_submit_button("Submit Answers")
    if submit:
        st.session_state["answers"] = answers
        # Grade
        total_score = 0
        max_score = 0
        feedback_list = []
        with st.spinner("Grading..."):
            for q in quiz:
                user_ans = answers.get(str(q.id), "")
                if q.type == "multiple_choice":
                    max_score += 1
                    is_correct = False
                    # compare strings normalized
                    if isinstance(q.answer, str) and user_ans.strip().lower() == q.answer.strip().lower():
                        is_correct = True
                    # Some generated quizzes might store correct choice as an index like "B" or "1" - try to be forgiving
                    # If choices list exists, try to map index
                    if not is_correct and q.choices:
                        # try index matching (1-based)
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
                    # open answer: grade with LLM or fallback
                    max_score += 100
                    if GPT4ALL_AVAILABLE and model_name:
                        grade = grade_open_answer_with_gpt4all(model_name, q.prompt, q.answer, user_ans, difficulty)
                    elif GPT4ALL_AVAILABLE:
                        grade = grade_open_answer_with_gpt4all("gpt4all-lora-quantized.bin", q.prompt, q.answer, user_ans, difficulty)
                    else:
                        grade = grade_open_answer_with_gpt4all("", q.prompt, q.answer, user_ans, difficulty)
                    # Normalize
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
        # compute percent for multiple-choice (where scale differs)
        # For simplicity, compute an overall percent:
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

# Footer / tips
st.markdown("----")
st.write("Tips: For best results with detailed, accurate questions and grading, install a local model (e.g., GPT4All) and specify the model filename in the sidebar. If you don't have a model, the app will use a simple fallback generator and grading heuristic.")
