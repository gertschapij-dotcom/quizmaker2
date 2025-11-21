```markdown
# Quiz Generator — Streamlit (Google Generative API optional)

This Streamlit app generates quizzes from either a large context text file or a short subject line.
It supports options for difficulty, number of questions, and question type (multiple choice, open answer, or mix).
You can provide a Google Generative API key (for example to use text-bison-001) to enable higher-quality generation and grading.
If you don't provide a key, the app falls back to a simple built-in generator and naive grader so the app remains fully functional offline.

Files:
- streamlit_app.py — the complete single-file Streamlit application.
- requirements.txt — pip installable dependencies.

Quick start:
1. Create a virtual environment (recommended) and activate it.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run streamlit_app.py
4. (Optional) Paste your Google API key in the sidebar. The key is taken as input in the sidebar (password field) and used for calls to the Generative API.

Google usage notes:
- This app sends prompts to the Google Generative API endpoint when you provide an API key.
- Example model name: text-bison-001
- If you do not have a key, leave the field blank and the app will use the fallback generator.
- Do not paste API keys you don't trust. Keep keys private.

Security & privacy:
- The API key is kept only in your running Streamlit session. This app does not store keys on disk.
- If you deploy the app to a public server, be careful to protect keys and access.

Extending the app:
- Swap in a different hosted LLM provider or add more robust parsing and retries.
- Improve prompt engineering to force cleaner JSON output from the model.
- Add export of results, timers, or more question types.

License: example code — adapt and improve as needed.
```
