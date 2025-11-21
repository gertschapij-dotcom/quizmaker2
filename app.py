# AI Quiz Generator — Streamlit Webapp (FREE, NO API KEYS)
# Save this file as app.py and run with: streamlit run app.py
# Updated to generate multiple (default 10) questions and to improve open-question prompts + post-processing.

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
import sys
import traceback
import re

# -----------------------------------------------------------
# LOAD MODELS (FREE OPEN SOURCE)
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    # spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            nlp = None
            st.warning(f"spaCy model could not be loaded: {e}")

    # Prefer instruction-tuned models (text2text) first
    text_pipeline = None
    text_pipeline_task = None
    text_model = None
    tried_models = []

    model_candidates = [
        ("google/flan-t5-small", "text2text-generation"),
        ("google/flan-t5-base", "text2text-generation"),
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
    ]

    for model_name, task in model_candidates:
        try:
            text_pipeline = pipeline(task, model=model_name)
            text_pipeline_task = task
            text_model = model_name
            break
        except Exception as e:
            tried_models.append((model_name, str(e)))
            text_pipeline = None

    if text_pipeline is None:
        st.warning(
            "No text-generation model could be loaded. The app will use a simple fallback generator."
        )

    # Sentence-transformers for semantic similarity (feedback)
    feedback_model = None
    try:
        feedback_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Could not load SentenceTransformer model: {e}")
        feedback_model = None

    return nlp, text_pipeline, feedback_model, text_pipeline_task


nlp, quiz_model, feedback_model, quiz_model_task = load_models()


# -----------------------------------------------------------
# GENERATION HELPERS (deterministic for instruction models)
# -----------------------------------------------------------
def generate_text(prompt, max_new_tokens=200):
    if quiz_model is not None:
        try:
            # If the pipeline is a text2text/instruction model (T5/Flan), prefer deterministic beams
            if quiz_model_task == "text2text-generation":
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "num_beams": 4,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3,
                }
            else:
                # Autoregressive fallback: low temperature sampling to reduce gibberish
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "no_repeat_ngram_size": 3,
                }

            out = quiz_model(prompt, **gen_kwargs)

            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if "generated_text" in first:
                    return first["generated_text"].strip()
                elif "text" in first:
                    return first["text"].strip()
                else:
                    return str(first).strip()
            return str(out).strip()
        except Exception:
            traceback_str = traceback.format_exc()
            st.error("Text-generation model crashed while generating text. See trace below.")
            st.text_area("Model error trace", traceback_str, height=200)
            return fallback_generator(prompt, max_new_tokens)
    else:
        return fallback_generator(prompt, max_new_tokens)


def fallback_generator(prompt, max_new_tokens):
    amount = 10
    for token in prompt.split():
        if token.isdigit():
            try:
                amount = int(token)
                break
            except Exception:
                pass

    if "meerkeuze" in prompt.lower() or "multiple choice" in prompt.lower():
        out_lines = []
        for i in range(amount):
            out_lines.append(f"{i+1}. Vraag over onderwerp:")
            out_lines.append("A) Juist antwoord")
            out_lines.append("B) Fout antwoord 1")
            out_lines.append("C) Fout antwoord 2")
            out_lines.append("D) Fout antwoord 3")
            out_lines.append("")
        return "\n".join(out_lines)
    else:
        out_lines = []
        for i in range(amount):
            out_lines.append(f"{i+1}. Noem één belangrijk punt over het onderwerp.")
        return "\n".join(out_lines)


# -----------------------------------------------------------
# PROMPTS — explicit, with example and strict output instruction
# -----------------------------------------------------------
def make_mc_questions(text, amount=10):
    example = (
        "Voorbeeld:\n"
        "Tekst: Nederland heeft een monetaire unie en belangrijke export.\n"
        "Antwoord:\n"
        "1. Welke stad is de hoofdstad van Nederland?\n"
        "A) Amsterdam\nB) Rotterdam\nC) Den Haag\nD) Utrecht\n\n"
    )

    prompt = (
        "Opdracht: Maak multiple choice-vragen.\n"
        "Output: Geef alleen de vragen en opties. Herhaal of kopieer de instructie NIET.\n"
        "Format: Nummer. Vraag\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\n\n"
        f"{example}"
        f"Invoer: Maak {amount} meerkeuzevragen over deze tekst of dit onderwerp. "
        "Voor elke vraag: 1 juist antwoord en 3 foute antwoorden. Gebruik duidelijke nummering.\n\n"
        f"Tekst:\n{text}\n\nAntwoord:"
    )
    # Increase token limit to allow many MC questions
    out = generate_text(prompt, max_new_tokens=600)
    cleaned = post_process_output(out, amount)
    return cleaned


def make_open_questions(text, amount=10):
    # Few-shot example showing multiple numbered open questions
    example = (
        "Voorbeeld:\n"
        "Tekst: De industriële revolutie bracht grote technologische veranderingen.\n"
        "Antwoord:\n"
        "1. Noem een technische innovatie uit de industriële revolutie.\n"
        "2. Leg kort uit waarom urbanisatie toen toenam.\n"
        "3. Welke sectoren profiteerden het meest van mechanisatie?\n\n"
    )

    prompt = (
        "Opdracht: Maak open vragen.\n"
        "Output: Geef alleen de genummerde lijst met vragen. Herhaal of kopieer de instructie NIET.\n"
        "Format: Nummer. Vraag\\n\n"
        f"{example}"
        f"Invoer: Maak {amount} open toetsvragen gebaseerd op deze tekst of dit onderwerp. "
        "Geef GEEN antwoorden, alleen de vragen. Gebruik duidelijke nummering.\n\n"
        f"Tekst:\n{text}\n\nAntwoord:"
    )
    # Increase token limit for 10 open questions
    out = generate_text(prompt, max_new_tokens=400)
    cleaned = post_process_output(out, amount)
    return cleaned


def post_process_output(out, expected_amount=10):
    """
    Try to remove instruction echoing and return only the numbered questions block.
    We search for the first numbered item like '1.' and return from there.
    Also try to ensure we return up to expected_amount numbered items.
    """
    if not out:
        return out

    # Normalize some common artifacts
    out = out.replace("\r\n", "\n").strip()

    # find the first occurrence of a numbered question like '1.' or '1)'
    m = re.search(r'(^|\n)\s*1[.)]\s*', out)
    start_idx = m.start() if m else 0
    candidate = out[start_idx:].strip()

    # Split into lines and collect lines until we've seen expected_amount question numbers
    lines = candidate.splitlines()
    collected = []
    qcount = 0
    current_block = []
    for line in lines:
        if re.match(r'^\s*\d+[.)]\s+', line):
            # new question starts
            if current_block:
                collected.append("\n".join(current_block).strip())
            current_block = [line.strip()]
            qcount += 1
        else:
            # continuation line (e.g., options or multi-line question)
            if current_block:
                current_block.append(line.rstrip())
        if qcount >= expected_amount:
            # after capturing expected_amount questions, continue collecting their option lines (if any),
            # but stop once we've likely passed them (heuristic: break if we hit a blank line after options)
            # we'll continue to finish the current_block then break
            pass
    if current_block:
        collected.append("\n".join(current_block).strip())

    # If we found numbered questions, join the first expected_amount of them
    if collected and len(collected) >= 1:
        selected = collected[:expected_amount]
        # If the model included options (e.g., A/B/C/D) they are preserved; otherwise it's plain questions.
        return "\n\n".join(selected).strip()

    # fallback: try to extract lines starting with A)/A) pattern (for MC), or just return whole cleaned string
    return candidate


# -----------------------------------------------------------
# SEMANTIC FEEDBACK FOR OPEN ANSWERS
# -----------------------------------------------------------
def check_answer(correct_answer, user_answer):
    if feedback_model is None:
        return "Feedback-model niet beschikbaar. Installeer 'sentence-transformers' en herstart de app."

    try:
        emb1 = feedback_model.encode(correct_answer, convert_to_tensor=True)
        emb2 = feedback_model.encode(user_answer, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return "Kon de overeenkomst niet berekenen."

    if score > 0.85:
        return f"✅ Goed! (similarity: {score:.2f})"
    elif score > 0.65:
        return f"⚠️ Bijna goed — je mist wat details. (similarity: {score:.2f})"
    else:
        return f"❌ Niet correct — grote verschillen. (similarity: {score:.2f})"


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="AI Quiz Generator", layout="centered")
st.title("🤖 AI Quiz Generator (Gratis Versie)")
st.write("Maak automatisch quizvragen op basis van lesmateriaal **óf** een onderwerp.")

mode = st.radio(
    "Kies een modus:",
    ["Quiz uit lesmateriaal", "Quiz over een onderwerp (zonder tekst)"]
)

# Let the user choose how many questions to generate (default 10)
amount = st.slider("Aantal vragen", min_value=1, max_value=20, value=10, step=1)

# MODE A — TEXT INPUT
if mode == "Quiz uit lesmateriaal":
    text = st.text_area("Plak je lesmateriaal hieronder:")

    if st.button("Genereer quiz"):
        if text.strip() == "":
            st.warning("Voer eerst lesmateriaal in.")
        else:
            st.subheader("Meerkeuzevragen")
            with st.spinner("Genereer meerkeuzevragen..."):
                st.write(make_mc_questions(text, amount=amount))

            st.subheader("Open vragen")
            with st.spinner("Genereer open vragen..."):
                st.write(make_open_questions(text, amount=amount))


# MODE B — QUIZ WITHOUT TEXT
if mode == "Quiz over een onderwerp (zonder tekst)":
    topic = st.text_input("Voer een onderwerp in (bijv. Biologie, WO2, Python).")

    if st.button("Genereer quiz over onderwerp"):
        if topic.strip() == "":
            st.warning("Voer een onderwerp in.")
        else:
            st.subheader(f"Quiz over: {topic}")
            with st.spinner("Genereer meerkeuzevragen..."):
                st.write(make_mc_questions(topic, amount=amount))

            st.subheader("Open vragen")
            with st.spinner("Genereer open vragen..."):
                st.write(make_open_questions(topic, amount=amount))


# FEEDBACK SECTION
st.divider()
st.header("📝 Antwoord Feedback Checker (Open Vragen)")

correct = st.text_input("Modelantwoord (juiste antwoord):")
user = st.text_input("Jouw antwoord:")

if st.button("Geef feedback"):
    if correct.strip() == "" or user.strip() == "":
        st.warning("Vul beide velden in.")
    else:
        with st.spinner("Controleer antwoord..."):
            st.write(check_answer(correct, user))

st.divider()
st.caption("Made with ❤️ — volledig gratis, open-source modellen.")
