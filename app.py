# AI Quiz Generator — Streamlit Webapp (FREE, NO API KEYS)
# Save this file as app.py and run with: streamlit run app.py
# If you previously saved the file with the name "code" (no .py extension)
# Streamlit will complain "not a valid python script". Rename the file to app.py.

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
            # If spaCy fails, we continue but set nlp to None
            nlp = None
            st.warning(f"spaCy model could not be loaded: {e}")

    # Text-generation model
    # NOTE: the originally suggested model (HuggingFaceH4/zephyr-7b-alpha) is very large and
    # may fail on machines without sufficient memory or GPU.
    # We'll try a smaller default model that is more likely to run on CPU.
    text_model = None
    text_pipeline = None
    tried_models = []
    for model_name in ["distilgpt2", "gpt2", "HuggingFaceH4/zephyr-7b-alpha"]:
        try:
            text_pipeline = pipeline("text-generation", model=model_name)
            text_model = model_name
            break
        except Exception as e:
            tried_models.append((model_name, str(e)))
            text_pipeline = None

    if text_pipeline is None:
        st.warning("No text-generation model could be loaded. The app will use a simple fallback generator.")
        # keep text_pipeline as None and handle fallback elsewhere

    # Sentence-transformers for semantic similarity (feedback)
    feedback_model = None
    try:
        feedback_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Could not load SentenceTransformer model: {e}")
        feedback_model = None

    return nlp, text_pipeline, feedback_model


nlp, quiz_model, feedback_model = load_models()


# Wrapper to generate text consistently whether the pipeline loaded or not.
def generate_text(prompt, max_new_tokens=200):
    if quiz_model is not None:
        try:
            out = quiz_model(prompt, max_new_tokens=max_new_tokens)
            # models return a list of dicts with 'generated_text'
            if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                return out[0]["generated_text"].strip()
            # fallback if the structure is different
            return str(out)
        except Exception:
            # catch runtime errors from transformers
            traceback_str = traceback.format_exc()
            st.error("Text-generation model crashed while generating text. See logs.")
            st.text_area("Model error trace", traceback_str, height=200)
            return fallback_generator(prompt, max_new_tokens)
    else:
        return fallback_generator(prompt, max_new_tokens)


def fallback_generator(prompt, max_new_tokens):
    # Simple, deterministic fallback so the app still demonstrates output
    # This produces readable (but simple) questions if no heavy model is available.
    # It inspects the prompt to decide whether to create MC or open questions.
    lines = prompt.strip().splitlines()
    # try to find amount in prompt (simple heuristic)
    amount = 4
    for token in prompt.split():
        if token.isdigit():
            amount = int(token)
            break

    # If the word "meerkeuze" or "multiple choice" is in prompt, make MC-style output
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
        # open questions fallback
        out_lines = []
        for i in range(amount):
            out_lines.append(f"{i+1}. Noem één belangrijk punt over het onderwerp.")
        return "\n".join(out_lines)


# -----------------------------------------------------------
# QUESTION GENERATION FUNCTIONS
# -----------------------------------------------------------
def make_mc_questions(text, amount=4):
    prompt = f"""
Maak {amount} meerkeuzevragen over deze tekst OF over dit onderwerp.
Zorg bij elke vraag voor:
- 1 juist antwoord
- 3 foute antwoorden
- duidelijke nummering

Invoer:
{text}
"""
    out = generate_text(prompt, max_new_tokens=300)
    return out.strip()


def make_open_questions(text, amount=4):
    prompt = f"""
Maak {amount} open toetsvragen gebaseerd op deze tekst OF dit onderwerp.
Geef GEEN antwoorden. Alleen de vragen.

Invoer:
{text}
"""
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
