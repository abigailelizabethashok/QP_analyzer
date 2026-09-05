from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import json
import time

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# =========================================================
# MODELS
# =========================================================

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# AI ANALYSIS
# =========================================================

def analyze_question_paper(question_paper, syllabus):

    prompt = f"""
You are an expert university question paper analyzer.

Analyze the following question paper using the provided syllabus.

========================
SYLLABUS
========================

{syllabus}


========================
QUESTION PAPER
========================

{question_paper}


Analyze the paper based on:

1. SYLLABUS COVERAGE

Find:
- Topics covered
- Topics missing
- Module-wise coverage
- Approximate coverage percentage


2. DIFFICULTY

Classify questions as:

Easy
Medium
Hard

Calculate approximate percentages.


3. REPETITION

Find:
- Repeated questions
- Repeated concepts
- Very similar questions
- Overused topics


4. MARKS DISTRIBUTION

Analyze:
- Marks per question
- Total marks
- Module-wise marks
- Whether marks are balanced


5. BLOOM'S TAXONOMY

Classify questions into:

Remember
Understand
Apply
Analyze
Evaluate
Create

Calculate approximate percentages.


6. PROBLEMS

Find issues such as:

- Too many questions from one module
- Missing topics
- Repeated questions
- Too many easy questions
- Too many difficult questions
- Poor marks distribution
- Poor Bloom's balance
- Lack of application questions


7. SUGGESTIONS

Give practical suggestions for the teacher.

Examples:

- Add questions from missing topics
- Replace repeated questions
- Add application questions
- Add analysis questions
- Improve difficulty balance
- Improve marks distribution


IMPORTANT:

Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_score": 0,

    "summary": "",

    "syllabus_coverage": {{
        "covered_topics": [],
        "missing_topics": [],
        "coverage_percentage": 0
    }},

    "difficulty": {{
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "comment": ""
    }},

    "repetition": {{
        "repeated_questions": [],
        "repeated_concepts": [],
        "comment": ""
    }},

    "marks_distribution": {{
        "distribution": [],
        "comment": ""
    }},

    "blooms_taxonomy": {{
        "Remember": 0,
        "Understand": 0,
        "Apply": 0,
        "Analyze": 0,
        "Evaluate": 0,
        "Create": 0,
        "comment": ""
    }},

    "problems": [],

    "suggestions": []
}}
"""

    last_error = ""

    # Try different models automatically
    for model in MODELS:

        try:

            print("Trying model:", model)

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            result = response.text

            if not result:
                continue

            # Remove markdown JSON formatting if Gemini adds it
            result = result.replace("```json", "")
            result = result.replace("```", "")
            result = result.strip()

            # Find JSON if extra text exists
            start = result.find("{")
            end = result.rfind("}")

            if start != -1 and end != -1:

                result = result[start:end + 1]

                data = json.loads(result)

                print("SUCCESS! Model used:", model)

                return data

        except Exception as e:

            last_error = str(e)

            print("Model failed:", model)
            print("Reason:", last_error)

            # Wait briefly before trying next model
            time.sleep(1)

            continue


    # If every model fails
    raise Exception(
        "All Gemini models failed.\n\n"
        "Last error:\n" + last_error
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# ANALYZE
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        syllabus = request.form.get(
            "syllabus",
            ""
        )

        question_paper = request.form.get(
            "question_paper",
            ""
        )

        uploaded_file = request.files.get("file")


        # -------------------------
        # PDF
        # -------------------------

        if uploaded_file and uploaded_file.filename:

            if uploaded_file.filename.lower().endswith(".pdf"):

                question_paper = extract_pdf_text(
                    uploaded_file
                )

            else:

                return jsonify({
                    "error": "Please upload a PDF file."
                }), 400


        # -------------------------
        # VALIDATION
        # -------------------------

        if not question_paper.strip():

            return jsonify({
                "error":
                "Please paste a question paper or upload a PDF."
            }), 400


        if not syllabus.strip():

            return jsonify({
                "error":
                "Please enter the syllabus."
            }), 400


        # -------------------------
        # AI
        # -------------------------

        result = analyze_question_paper(
            question_paper,
            syllabus
        )


        return jsonify(result)


    except Exception as e:

        print("ERROR:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )