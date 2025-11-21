"""
AI-free Quiz Generator — Streamlit Webapp (Dutch)
This version is implemented from scratch with no heavy LLM requirements.
It uses spaCy to analyze text and generates:
 - multiple-choice questions (MC) by creating cloze (fill-in) items from named entities
 - open questions using templates and sentence extraction

How to run:
 - Save as app.py
 - Ensure dependencies: pip install streamlit spacy
 - Download spaCy model: python -m spacy download en_core_web_sm
 - Run: streamlit run app.py

Notes:
 - This approach avoids large transformers models so it runs reliably on Streamlit Cloud.
 - If sentence-transformers is installed, the app will use it for semantic feedback; otherwise a simple string-similarity fallback is used.
"""
import streamlit as st
import random
import re
from typing import List, Tuple

# Try to import spacy and sentence-transformers (optional)
try:
    import spacy
    from spacy.lang.nl import Dutch
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

# --- Utility data for distractors ---
COMMON_PERSONS = [
    "Jan", "Piet", "Karel", "Sofie", "Anna", "Maria", "Willem", "Kees", "Jeroen", "Lisa"
]
COMMON_ORGS = [
    "De Nederlandse Bank", "Ministerie van Financiën", "EU", "IMF", "World Bank",
    "Bundesbank", "BMW", "Deutsche Bank"
]
COMMON_GPE = [
    "Nederland", "Duitsland", "België", "Frankrijk", "Berlijn", "München", "Hamburg", "Brussel"
]
COMMON_TOPICS = [
    "economie", "geschiedenis", "biologie", "programmeren", "wiskunde", "aardrijkskunde"
]

# --- Load spaCy if available, try to download model if not loaded ---
nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Try to download the model (best-effort)
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

# --- Optional sentence-transformers model for semantic feedback ---
st_model = None
if ST_AVAILABLE:
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        st_model = None

# --- Core generation logic ---

def sanitize_text(text: str) -> str:
    return text.strip()

def split_sentences(text: str) -> List[str]:
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # fallback: naive split on punctuation
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def extract_entities(text: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (ent_text, ent_label, sentence) from the text
    Uses spaCy NER when available; otherwise heuristics (dates/numbers/caps)
    """
    entities = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sents:
            for ent in sent.ents:
                entities.append((ent.text, ent.label_, sent.text.strip()))
    else:
        # Very simple heuristic: look for capitalized phrases and years/dates
        sents = split_sentences(text)
        for s in sents:
            caps = re.findall(r'\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*', s)
            for c in caps:
                entities.append((c, "PROPN", s))
            years = re.findall(r'\b(18|19|20)\d{2}\b', s)
            for y in years:
                entities.append((y, "DATE", s))
    return entities

def make_cloze_from_entity(ent_text: str, sentence: str) -> str:
    """
    Replace the first occurrence of ent_text in sentence with a blank
    """
    # escape regex chars in ent_text
    pattern = re.escape(ent_text)
    new = re.sub(pattern, "_____", sentence, count=1, flags=re.IGNORECASE)
    # If replacement failed, append blank at end
    if "_____" not in new:
        return sentence + " _____"
    return new

def generate_distractors(correct: str, label: str, pool: List[str]) -> List[str]:
    """
    Generate up to 3 distractors based on label and pool of found entities.
    Always returns 3 items (may include simple perturbations).
    """
    distractors = []
    pool_candidates = [p for p in pool if p.lower() != correct.lower()]
    random.shuffle(pool_candidates)

    # Use pool entities first
    for p in pool_candidates:
        if len(distractors) >= 3:
            break
        # simple sanity: avoid identical short tokens
        if p.strip() and p.lower() != correct.lower():
            distractors.append(p)

    # If not enough, use label-based common lists or perturbations
    if len(distractors) < 3:
        if label in ("PERSON", "PER", "PROPN"):
            src = COMMON_PERSONS
            for v in src:
                if v.lower() != correct.lower() and v not in distractors:
                    distractors.append(v)
                    if len(distractors) >= 3:
                        break
        elif label in ("ORG", "ORG_"):
            for v in COMMON_ORGS:
                if v.lower() != correct.lower() and v not in distractors:
                    distractors.append(v)
                    if len(distractors) >= 3:
                        break
        elif label in ("GPE", "LOC"):
            for v in COMMON_GPE:
                if v.lower() != correct.lower() and v not in distractors:
                    distractors.append(v)
                    if len(distractors) >= 3:
                        break
        elif label in ("DATE",):
            # perturb years if possible
            try:
                y = int(re.search(r'\d{4}', correct).group())
                for delta in (1, -1, 5, -5, 10):
                    candidate = str(y + delta)
                    if candidate not in distractors and candidate != correct:
                        distractors.append(candidate)
                        if len(distractors) >= 3:
                            break
            except Exception:
                pass

    # Final fallback: generate simple variations
    while len(distractors) < 3:
        # add a mild corruption
        corrupted = corrupt_string(correct)
        if corrupted not in distractors and corrupted.lower() != correct.lower():
            distractors.append(corrupted)
        else:
            # pick from common topics
            x = random.choice(COMMON_TOPICS)
            if x not in distractors:
                distractors.append(x)

    return distractors[:3]

def corrupt_string(s: str) -> str:
    if not s:
        return "Optie"
    s = s.strip()
    # swap two letters if possible
    if len(s) > 3:
        i = random.randint(0, len(s) - 2)
        lst = list(s)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        return "".join(lst)
    # else add suffix
    return s + random.choice(["a", "en", "en", "s"])

# --- Public generation functions ---

def generate_multiple_choice(text: str, amount: int = 10) -> str:
    """
    Generate 'amount' MC questions from text. Returns a formatted string.
    Strategy:
     - Extract entities and sentences
     - For each entity create a cloze question and 3 distractors
     - If not enough entities, create topical MC items from keywords/templates
    """
    text = sanitize_text(text)
    sents = split_sentences(text)
    entities = extract_entities(text)

    # Build pool for distractors
    pool = [e[0] for e in entities]

    questions = []
    used = set()

    # Prioritize entities with longer text (more informative) and sentence length
    entities_sorted = sorted(entities, key=lambda x: (-len(x[0]), -len(x[2])))

    for ent_text, ent_label, sentence in entities_sorted:
        if len(questions) >= amount:
            break
        key_lower = ent_text.lower()
        if key_lower in used or len(ent_text.strip()) <= 1:
            continue
        used.add(key_lower)
        cloze = make_cloze_from_entity(ent_text, sentence)
        correct = ent_text.strip()
        distractors = generate_distractors(correct, ent_label, pool)
        # shuffle options and keep track of answer
        opts = [correct] + distractors
        random.shuffle(opts)
        opt_letters = ["A", "B", "C", "D"]
        options_text = "\n".join([f"{opt_letters[i]}) {opts[i]}" for i in range(4)])
        # find correct letter
        correct_letter = opt_letters[opts.index(correct)]
        q_text = f"{len(questions)+1}. {cloze}\n{options_text}\nAntwoord: {correct_letter}) {correct}"
        questions.append(q_text)

    # If not enough questions produced, generate template questions based on topic phrases
    if len(questions) < amount:
        # Generate based on keywords (nouns)
        words = re.findall(r'\b[A-Za-zÀ-ÿ\-]{4,}\b', text)
        keywords = []
        for w in words:
            if w.lower() not in COMMON_TOPICS and w.isalpha():
                keywords.append(w)
        keywords = list(dict.fromkeys(keywords))  # unique preserve order
        i = 0
        while len(questions) < amount:
            keyword = keywords[i % max(1, len(keywords))] if keywords else random.choice(COMMON_TOPICS)
            template_sentence = f"Welk van de volgende beweringen hoort bij {keyword}?"
            correct = keyword
            distractors = generate_distractors(correct, "PROPN", pool)
            opts = [correct] + distractors
            random.shuffle(opts)
            opt_letters = ["A", "B", "C", "D"]
            options_text = "\n".join([f"{opt_letters[j]}) {opts[j]}" for j in range(4)])
            q_text = f"{len(questions)+1}. {template_sentence}\n{options_text}\nAntwoord: {opt_letters[opts.index(correct)])}) {correct}"
            # Fix double parentheses if any formatting oddities
            q_text = q_text.replace("))", ")")
            questions.append(q_text)
            i += 1

    return "\n\n".join(questions[:amount])


def generate_open_questions(text: str, amount: int = 10) -> str:
    """
    Generate 'amount' open questions:
     - Prefer cloze-to-question conversions (replace named entity with 'Vul in' style)
     - If topic-only (short input), use templates to produce many open questions
    """
    text = sanitize_text(text)
    sents = split_sentences(text)
    entities = extract_entities(text)

    questions = []
    # 1) From sentences with entities, produce 'Vul in' or 'Leg uit' variants
    for ent_text, ent_label, sentence in entities:
        if len(questions) >= amount:
            break
        cloze = make_cloze_from_entity(ent_text, sentence)
        # Turn into a question form: either fill-in or explain
        if ent_label in ("PERSON", "ORG", "GPE", "PROPN"):
            q = f"{len(questions)+1}. Vul in: {cloze}"
        elif ent_label in ("DATE",):
            q = f"{len(questions)+1}. Wanneer gebeurde het volgende: {cloze}"
        else:
            q = f"{len(questions)+1}. Leg uit: {sentence}"
        questions.append(q)

    # 2) If not enough, create template open questions from main sentences
    i = 0
    for sent in sents:
        if len(questions) >= amount:
            break
        # skip very short sentences
        if len(sent.split()) < 5:
            continue
        # create question templates
        templates = [
            "{}. Vat de volgende zin samen in 1-2 zinnen: {}",
            "{}. Noem twee belangrijke punten uit de volgende zin: {}",
            "{}. Stel een toetsvraag over de volgende zin: {}",
            "{}. Leg in het kort uit waarom dit belangrijk is: {}"
        ]
        tmpl = templates[i % len(templates)]
        q = tmpl.format(len(questions)+1, sent)
        questions.append(q)
        i += 1

    # 3) If still not enough (e.g., short input or topic-only), use topic templates
    if len(questions) < amount:
        base = text if len(text.split()) > 1 else text or "dit onderwerp"
        topic_templates = [
            "{}. Beschrijf in eigen woorden wat '{}' inhoudt.",
            "{}. Noem drie mogelijke oorzaken van '{}'.",
            "{}. Welke gevolgen kan '{}' hebben op korte termijn?",
            "{}. Welke rol spelen belangrijke actoren in '{}'?"
        ]
        j = 0
        while len(questions) < amount:
            tmpl = topic_templates[j % len(topic_templates)]
            q = tmpl.format(len(questions)+1, base)
            questions.append(q)
            j += 1

    # Ensure we return exactly 'amount' questions
    return "\n\n".join(questions[:amount])

# --- Simple feedback checker (uses sentence-transformers if available) ---
def check_answer(correct: str, user: str) -> str:
    if st_model:
        try:
            emb1 = st_model.encode(correct, convert_to_tensor=True)
            emb2 = st_model.encode(user, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item()
            if score > 0.8:
                return f"✅ Goed — similarity {score:.2f}"
            elif score > 0.55:
                return f"⚠️ Redelijk — similarity {score:.2f}"
            else:
                return f"❌ Niet correct — similarity {score:.2f}"
        except Exception:
            pass
    # fallback: simple substring / ratio
    corr = re.sub(r'\W+', ' ', correct.lower()).strip()
    us = re.sub(r'\W+', ' ', user.lower()).strip()
    if not corr or not us:
        return "Vul beide velden in."
    if corr in us or us in corr:
        return "✅ Goed (string match)"
    # simple token overlap
    corr_set = set(corr.split())
    us_set = set(us.split())
    if not corr_set:
        return "Geen referentietekst."
    overlap = len(corr_set & us_set) / max(1, len(corr_set))
    if overlap > 0.5:
        return f"⚠️ Gedeeltelijk goed (overlap {overlap:.2f})"
    return f"❌ Niet correct (overlap {overlap:.2f})"

# --- Streamlit UI ---
st.set_page_config(page_title="Quiz Generator (LLM-free)", layout="centered")
st.title("Quiz Generator — LLM-free (SpaCy)")

st.write("Genereer multiple-choice en open vragen gebaseerd op tekst of een onderwerp. "
         "Deze versie werkt zonder grote LLMs en gebruikt spaCy voor analyse.")

mode = st.radio("Kies modus:", ["Quiz uit lesmateriaal", "Quiz over een onderwerp (zonder tekst)"])

amount = st.slider("Aantal vragen", min_value=1, max_value=20, value=10, step=1)

if mode == "Quiz uit lesmateriaal":
    text = st.text_area("Plak je lesmateriaal hier:", height=250)
    if st.button("Genereer quiz"):
        if not text.strip():
            st.warning("Voer eerst lesmateriaal in.")
        else:
            with st.spinner("Genereer multiple-choice..."):
                mc = generate_multiple_choice(text, amount=amount)
                st.subheader("Meerkeuzevragen")
                st.text_area("Meerkeuzevragen", value=mc, height=300)
            with st.spinner("Genereer open vragen..."):
                oq = generate_open_questions(text, amount=amount)
                st.subheader("Open vragen")
                st.text_area("Open vragen", value=oq, height=300)

else:
    topic = st.text_input("Voer een onderwerp in (bijv. Biologie, WO2, Python):")
    if st.button("Genereer quiz over onderwerp"):
        if not topic.strip():
            st.warning("Voer een onderwerp in.")
        else:
            # For topic-only, create synthetic text to extract entities/keywords
            synthetic_text = (
                f"{topic}. {topic} is een belangrijk onderwerp dat vaak wordt behandeld in lessen. "
                f"Belangrijke aspecten van {topic} zijn beleid, geschiedenis, theorieën en voorbeelden."
            )
            with st.spinner("Genereer multiple-choice..."):
                mc = generate_multiple_choice(synthetic_text, amount=amount)
                st.subheader(f"Meerkeuzevragen — {topic}")
                st.text_area("Meerkeuzevragen", value=mc, height=300)
            with st.spinner("Genereer open vragen..."):
                oq = generate_open_questions(topic, amount=amount)
                st.subheader("Open vragen")
                st.text_area("Open vragen", value=oq, height=300)

st.divider()
st.header("📝 Antwoord Feedback Checker (Open Vragen)")
correct = st.text_input("Modelantwoord (juiste antwoord):")
user = st.text_input("Jouw antwoord:")
if st.button("Geef feedback"):
    res = check_answer(correct, user)
    st.write(res)

st.caption("Deze versie gebruikt heuristieken en spaCy om relevante vragen te genereren zonder een LLM. "
           "Als je wilt, kan ik een version maken die uses a small instruction-tuned model for better quality (requires additional dependencies).")
