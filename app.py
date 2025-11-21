# AI Quiz Generator — Streamlit Webapp (FREE, NO API KEYS)
# Save this file as app.py and run with: streamlit run app.py
# This version prefers instruction-tuned models (Flan-T5) and uses clearer prompts
# to avoid the model echoing the instructions or producing gibberish.

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
import sys
import traceback

# -----------------------------------------------------------
# LOAD MODELS (FREE OPEN SOURCE) — robust/fallback behavior
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
        # fallback to autoregressive models if T5 not available
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
    ]

    for model_name, task in model_candidates:
        try:
            # create pipeline for the task
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


# Unpack loaded models (quiz_model may be None if none could be loaded)
nlp, quiz_model, feedback_model, quiz_model_task = load_models()


# -----------------------------------------------------------
# GENERATION HELPERS
# -----------------------------------------------------------
def generate_text(prompt, max_new_tokens=200):
    """
    Use the loaded pipeline to generate text. If no pipeline is available,
    fall back to a simple deterministic generator so the app still shows output.
    """
    if quiz_model is not None:
        try:
            # Use conservative generation settings to reduce repetition/gibberish.
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.25,  # low temp -> less random nonsense
                "top_p": 0.9,
            }
            out = quiz_model(prompt, **gen_kwargs)
            # pipeline returns a list of dicts with 'generated_text'
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                # Some pipelines (different versions) may return 'generated_text' or 'text'
                if "generated_text" in first:
                    return first["generated_text"].strip()
                elif "text" in first:
                    return first["text"].strip()
                else:
                    return str(first)
            return str(out).strip()
        except Exception:
            # Show the traceback in the Streamlit UI so you can debug in logs
            traceback_str = traceback.format_exc()
            st.error("Text-generation model crashed while generating text. See trace below.")
            st.text_area("Model error trace", traceback_str, height=200)
            return fallback_generator(prompt, max_new_tokens)
    else:
        return fallback_generator(prompt, max_new_tokens)


def fallback_generator(prompt, max_new_tokens):
    """
    Very simple deterministic fallback so the UI still shows plausible output
    when no model is available.
    """
    # try to find amount in prompt (simple heuristic)
    amount = 4
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
            out_lines.append("")  # blank line
        return "\n".join(out_lines)
    else:
        out_lines = []
        for i in range(amount):
            out_lines.append(f"{i+1}. Noem één belangrijk punt over het onderwerp.")
        return "\n".join(out_lines)


# -----------------------------------------------------------
# QUESTION GENERATION FUNCTIONS
# -----------------------------------------------------------
def make_mc_questions(text, amount=4):
    # Use a clear instruction-style prompt that T5 and other instruction models follow well.
    prompt = (
        "Opdracht: Maak multiple choice-vragen.\n"
        f"Invoer: Maak {amount} meerkeuzevragen over deze tekst of dit onderwerp. "
        "Voor elke vraag: 1 juist antwoord en 3 foute antwoorden. Gebruik duidelijke nummering.\n"
        f"Tekst:\n{text}\n\nAntwoord:"
    )
    out = generate_text(prompt, max_new_tokens=300)
    return out.strip()


def make_open_questions(text, amount=4):
    prompt = (
        "Opdracht: Maak open vragen.\n"
        f"Invoer: Maak {amount} open toetsvragen gebaseerd op deze tekst of dit onderwerp. "
        "Geef GEEN antwoorden, alleen de vragen. Gebruik duidelijke nummering.\n"
        f"Tekst:\n{text}\n\nAntwoord:"
    )
    out = generate_text(prompt, max_new_tokens=200)
    return out.strip()


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

# MODE A — TEXT INPUT
if mode == "Quiz uit lesmateriaal":
    text = st.text_area("Plak je lesmateriaal hieronder:")

    if st.button("Genereer quiz"):
        if text.strip() == "":
            st.warning("Voer eerst lesmateriaal in.")
        else:
            st.subheader("Meerkeuzevragen")
            with st.spinner("Genereer meerkeuzevragen..."):
                st.write(make_mc_questions(text))

            st.subheader("Open vragen")
            with st.spinner("Genereer open vragen..."):
                st.write(make_open_questions(text))


# MODE B — QUIZ WITHOUT TEXT
if mode == "Quiz over een onderwerp (zonder tekst)":
    topic = st.text_input("Voer een onderwerp in (bijv. Biologie, WO2, Python).")

    if st.button("Genereer quiz over onderwerp"):
        if topic.strip() == "":
            st.warning("Voer een onderwerp in.")
        else:
            st.subheader(f"Quiz over: {topic}")
            with st.spinner("Genereer meerkeuzevragen..."):
                st.write(make_mc_questions(topic))

            st.subheader("Open vragen")
            with st.spinner("Genereer open vragen..."):
                st.write(make_open_questions(topic))


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
