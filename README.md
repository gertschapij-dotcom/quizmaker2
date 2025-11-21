# quizmaker2
quizmaker Inholland

# Local LLM Quiz Generator — Streamlit

This Streamlit app creates quizzes from either a large context text file or a short subject line using a free self-contained local LLM (recommended: GPT4All). It supports options for difficulty, number of questions, and question types (multiple choice, open answer, or mix). After you submit answers, it provides detailed feedback — for open answers it uses the local model to grade and explain.

Files:
- streamlit_app.py — the Streamlit application.
- requirements.txt — Python packages needed.

Quick start (recommended with GPT4All):
1. Install Python 3.8+ and create a virtualenv.
2. Install dependencies:
   pip install -r requirements.txt
3. Install GPT4All and download a small local model:
   - pip install gpt4all
   - Download a compatible model from: https://gpt4all.io/models/
     Example models: `ggml-gpt4all-j.bin`, `gpt4all-lora-quantized.bin` (model sizes vary).
   - Place the model file in a known location and provide the filename in the app sidebar.
4. Run the app:
   streamlit run streamlit_app.py

Notes if you don't have a local model:
- The app contains a small fallback quiz generator and a naive grader so it will still work, but results will be simpler and less reliable than a real LLM-based generation.
- Providing a local model enables richer question generation and more meaningful open-answer grading.

Security & privacy:
- The app runs locally and uses a local LLM (if you provide one). No data is sent to external APIs in that configuration.
- If you use an external model or expose your app, consider privacy implications.

Troubleshooting:
- If gpt4all import fails, make sure you installed the package in the same environment running Streamlit.
- If model loading fails, check the model filename and compatibility with the gpt4all binary/package.

License: MIT-style (use at your own risk). This is example code intended to be adapted and improved for production needs.
