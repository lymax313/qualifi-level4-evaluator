from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from google import genai
from google.genai import types
import uvicorn
import os
import json
import re
from datetime import datetime
from fpdf import FPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Gemini client setup -----------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set – app will use fallback evaluations only")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- FastAPI app -------------------------------------------------------------

app = FastAPI(title="Qualifi Level 4 Evaluator", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

QUALIFI_RUBRIC = {
    "Content": 50,
    "Application of Theory and Literature": 40,
    "Knowledge and Understanding": 50,
    "Presentation/Writing Skills": 40,
    "Referencing": 40,
}

GRADE_BANDS = [
    {"name": "Distinction", "min": 70, "max": 100},
    {"name": "Merit", "min": 60, "max": 69},
    {"name": "Pass", "min": 40, "max": 59},
    {"name": "FAIL", "min": 0, "max": 39},
]

UNIT_CRITERIA = {
    "AID 401": [
        "2.1 Explain trending AI application used in different industries.",
        "2.2 Discuss how these AI applications are implemented and the benefits to industry.",
        "3.1 Assess AI trending tools used within industry and the challenges faced in implementing these technologies.",
        "3.2 Discuss possible impact from such application of the trending AI tools.",
        "4.1 Describe society's future needs in respect of sustainability.",
        "4.2 Review how AI applications can be optimized to address society's needs responsibly and sustainably.",
    ],
    "AID 402": [
        "2.1 Calculate and interpret probabilities for various events and distributions.",
        "2.2 Explain and apply the concepts of random variables and their distributions.",
        "2.3 Conduct hypothesis testing and construct confidence intervals.",
        "3.1 Apply linear regression to business scenarios.",
        "3.2 Understand and apply derivatives in context.",
        "3.3 Formulate and solve optimization problems.",
        "4.1 Apply integrals in business contexts.",
        "4.2 Understand and apply computational complexity analysis.",
        "4.3 Use mathematical methods to analyze and interpret complex datasets.",
    ],
    "AID 403": [
        "2.1 Organise data in pandas and develop meaningful insights.",
        "2.2 Perform data visualisations to help understand the data.",
        "2.3 Discuss and explain the findings from the visualisations.",
        "3.1 Apply statistical methods to the data.",
        "3.2 Explain the meaning of the statistical findings.",
        "4.1 Compare and justify the chosen method of data analysis.",
        "4.2 Present the findings of the data analysis to a range of stakeholders.",
    ],
    "AID 404": [
        "2.1 Critically evaluate the different types of machine learning algorithms.",
        "2.2 Discuss the suitability of machine learning algorithms for different tasks.",
        "3.1 Implement a machine learning algorithm for a specific task.",
        "3.2 Optimise the performance of the machine learning algorithm.",
        "4.1 Evaluate the performance of the machine learning algorithm.",
        "4.2 Discuss the limitations of the machine learning algorithm.",
    ],
    "AID 405": [
        "2.1 Explain the concept of deep learning and its key components.",
        "2.2 Discuss the role of neural networks in deep learning.",
        "3.1 Explain the purpose and expected outcome of the neural network model.",
        "3.2 Select a suitable model for image classification and explain the choice.",
        "3.3 Perform deep learning modelling on the selected dataset.",
        "4.1 Evaluate the performance of the deep learning model using appropriate metrics.",
        "4.2 Discuss how the performance of the model can be improved.",
    ],
    "AID 406": [
        "2.1 Identify the key elements of ethical frameworks and how they are implemented.",
        "2.2 Explain the importance of ethics in AI development and deployment.",
        "3.1 Identify bias, privacy, and security issues and explain how these can be addressed.",
        "3.2 Explain the methods that can be used to mitigate bias and ensure privacy and security.",
        "4.1 Discuss the responsibilities of government and industry in ensuring ethical AI.",
        "4.2 Discuss the differences in how AI is governed globally.",
    ],
}

# --- Helpers -----------------------------------------------------------------


def parse_rubric_file(content: bytes, filename: str) -> str | None:
    """Parse rubric from uploaded file, focusing on summative parts."""
    try:
        rubric_text = content.decode("utf-8", errors="ignore")
        summative_pattern = r"SUMMATIVE\s+TASK.*?(?=FORMATIVE|$)"
        summative_match = re.search(
            summative_pattern, rubric_text, re.DOTALL | re.IGNORECASE
        )

        if summative_match:
            return summative_match.group()

        if "Learning Outcomes" in rubric_text and "Assessment Criteria" in rubric_text:
            return rubric_text

        return rubric_text if len(rubric_text.strip()) > 50 else None
    except Exception as e:
        logger.error(f"Rubric parsing error: {e}")
        return None


def calculate_grade(percentage: float) -> str:
    for band in GRADE_BANDS:
        if band["min"] <= percentage <= band["max"]:
            return band["name"]
    return "Ungraded"


def generate_fallback_evaluation(reason: str) -> dict:
    logger.error(f"Using fallback evaluation. Reason: {reason}")
    return {
        "mark_breakdown": {
            k: {"score": 20, "justification": "Fallback due to system error"}
            for k in QUALIFI_RUBRIC.keys()
        },
        "total_score": 100,
        "percentage": 45.5,
        "grade": "Pass",
        "criteria_feedback": [],
        "overall_feedback": "A system error occurred during AI evaluation. Please review manually.",
        "strengths": [],
        "improvements": [],
        "evaluation": {},
    }


async def evaluate_with_gemini(
    assignment_text: str, rubric_text: str | None, unit_title: str
) -> dict:
    """Call Gemini and return a structured evaluation."""
    if client is None:
        return generate_fallback_evaluation("Gemini client not initialised (no API key)")

    try:
        unit_code = "AID 401"
        for code in UNIT_CRITERIA.keys():
            if code in unit_title:
                unit_code = code
                break

        unit_specific_criteria = "\n".join(
            f"- {c}" for c in UNIT_CRITERIA.get(unit_code, [])
        )

        rubric_snippet = (rubric_text or "")[:2000]

        # Ask Gemini for JSON directly. [web:30][web:35][web:38]
        prompt = f"""You are an AI evaluator for the Qualifi Level 4 Diploma in Artificial Intelligence.
You must mark ONLY the SUMMATIVE TASK for the specified unit: {unit_title}.

You are given:
1) The unit assignment brief context: {rubric_snippet}
2) The specific Assessment Criteria for this summative task:
{unit_specific_criteria}
3) The generic Qualifi Mark Scheme (Total 220 points):
   - Content: 0–50
   - Application of Theory and Literature: 0–40
   - Knowledge and Understanding: 0–50
   - Presentation/Writing Skills: 0–40
   - Referencing: 0–40

Return ONLY valid JSON using this schema:
{{
  "unit_code": "{unit_code}",
  "mark_breakdown": {{
    "Content": {{"score": 0, "justification": ""}},
    "Application of Theory and Literature": {{"score": 0, "justification": ""}},
    "Knowledge and Understanding": {{"score": 0, "justification": ""}},
    "Presentation/Writing Skills": {{"score": 0, "justification": ""}},
    "Referencing": {{"score": 0, "justification": ""}}
  }},
  "criteria_feedback": [
    {{
      "criterion": "Criterion text",
      "achieved": true,
      "evidence": "Evidence from text",
      "improvement": "Actionable suggestion"
    }}
  ],
  "overall_feedback": "",
  "strengths": [],
  "improvements": []
}}

LEARNER_ANSWER:
{assignment_text[:6000]}
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),  # ask for JSON [web:35][web:38]
        )

        raw_text = (response.text or "").strip()
        logger.info(f"Gemini raw response (first 400 chars): {raw_text[:400]}")

        try:
            evaluation = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from Gemini: {e}")
            return generate_fallback_evaluation("Gemini JSON decode error")

        total_score = 0
        for criterion, max_score in QUALIFI_RUBRIC.items():
            if (
                "mark_breakdown" in evaluation
                and criterion in evaluation["mark_breakdown"]
            ):
                score = min(
                    evaluation["mark_breakdown"][criterion].get("score", 0), max_score
                )
                evaluation["mark_breakdown"][criterion]["score"] = score
                total_score += score

        evaluation["total_score"] = total_score
        evaluation["percentage"] = round((total_score / 220) * 100, 1)
        evaluation["grade"] = calculate_grade(evaluation["percentage"])

        evaluation.setdefault("criteria_feedback", [])
        evaluation.setdefault("overall_feedback", "")
        evaluation.setdefault("strengths", [])
        evaluation.setdefault("improvements", [])
        evaluation.setdefault("evaluation", {})

        return evaluation

    except Exception as e:
        logger.error(f"Gemini evaluation error: {e}")
        return generate_fallback_evaluation("Unexpected Gemini error")


def generate_pdf(student_name: str, student_id: str, unit_title: str, evaluation: dict) -> str:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "QUALIFI LEVEL 4 DIPLOMA IN AI - EVALUATION REPORT", 0, 1, "C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.cell(100, 7, f"Student Name: {student_name}", 0, 0)
    pdf.cell(0, 7, f"Student ID: {student_id}", 0, 1)
    pdf.cell(0, 7, f"Unit: {unit_title}", 0, 1)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(240, 240, 240)
    grade = evaluation.get("grade", "N/A")
    score = evaluation.get("total_score", 0)
    perc = evaluation.get("percentage", 0)
    pdf.cell(0, 12, f"FINAL GRADE: {grade} | SCORE: {score}/220 ({perc}%)", 1, 1, "C", True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. MARK SCHEME BREAKDOWN", 0, 1)
    pdf.set_font("Arial", "", 10)

    breakdown = evaluation.get("mark_breakdown", {})
    for crit, data in breakdown.items():
        max_s = QUALIFI_RUBRIC.get(crit, 0)
        s = data.get("score", 0)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, f"{crit} ({s}/{max_s})", 0, 1)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5, data.get("justification", "No justification provided."))
        pdf.ln(2)

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. ASSESSMENT CRITERIA FEEDBACK", 0, 1)

    for item in evaluation.get("criteria_feedback", []):
        pdf.set_font("Arial", "B", 10)
        crit_text = item.get("criterion", "N/A")
        status = "[ACHIEVED]" if item.get("achieved") else "[NOT ACHIEVED]"
        pdf.multi_cell(0, 6, f"{status} {crit_text}")
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(0, 5, f"Evidence: {item.get('evidence', 'N/A')}")
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5, f"Improvement: {item.get('improvement', 'N/A')}")
        pdf.ln(2)

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. OVERALL EVALUATION", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, evaluation.get("overall_feedback", "N/A"))

    pdf.ln(2)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Key Strengths:", 0, 1)
    pdf.set_font("Arial", "", 10)
    for s in evaluation.get("strengths", []):
        pdf.multi_cell(0, 5, f"- {s}")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Areas for Improvement:", 0, 1)
    pdf.set_font("Arial", "", 10)
    for i in evaluation.get("improvements", []):
        pdf.multi_cell(0, 5, f"- {i}")

    filename = f"tmp/qualifi-pdfs/{student_id}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf.output(filename)
    return filename

# --- Routes ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception:
        return "Qualifi Evaluator API Running"


@app.post("/api/evaluate")
async def evaluate_assignment(
    student_name: str = Form(...),
    student_id: str = Form(...),
    unit_title: str = Form(...),
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...),
):
    try:
        rubric_content = await rubric.read()
        rubric_text = parse_rubric_file(rubric_content, rubric.filename)

        assignment_content = await assignment.read()
        assignment_text = assignment_content.decode("utf-8", errors="ignore")

        evaluation = await evaluate_with_gemini(assignment_text, rubric_text, unit_title)
        pdf_path = generate_pdf(student_name, student_id, unit_title, evaluation)

        return JSONResponse(
            {
                "success": True,
                "student_name": student_name,
                "student_id": student_id,
                "unit_title": unit_title,
                "data": evaluation,
                "pdf_url": f"/download/{os.path.basename(pdf_path)}",
            }
        )
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = f"tmp/qualifi-pdfs/{filename}"
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
    )
