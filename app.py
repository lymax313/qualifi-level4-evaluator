from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
import os
import json
import re
from datetime import datetime
from fpdf import FPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def feedback_for_score(score: int, max_score: int, area: str) -> str:
    """Return a canned comment based on score band for a rubric area."""
    ratio = score / max_score if max_score else 0
    if ratio >= 0.8:
        return (
            f"In {area}, the learner demonstrates consistently strong performance with "
            "clear, relevant coverage aligned to the assessment expectations."
        )
    elif ratio >= 0.6:
        return (
            f"In {area}, the learner meets most expectations but would benefit from "
            "deeper analysis and more specific supporting examples."
        )
    elif ratio >= 0.4:
        return (
            f"In {area}, the learner shows partial achievement; key points are either "
            "missing or under‑developed and should be expanded."
        )
    else:
        return (
            f"In {area}, the learner provides limited evidence and needs substantial "
            "improvement to meet the required standard."
        )


def evaluate_locally(assignment_text: str, rubric_text: str | None, unit_title: str) -> dict:
    """
    Simple heuristic evaluator, fully local (no API):
    - Scores based on word count, basic keyword presence, and formatting.
    - Generates banded justifications, overall feedback, strengths, improvements.
    """
    text_lower = assignment_text.lower()
    words = assignment_text.split()
    word_count = len(words)

    # Content & knowledge – more words => higher scores (up to max)
    content_score = min(QUALIFI_RUBRIC["Content"], int(word_count / 40))
    knowledge_score = min(QUALIFI_RUBRIC["Knowledge and Understanding"], int(word_count / 40))

    # Application of theory – count basic theoretical keywords
    theory_keywords = ["theory", "model", "framework", "research", "study", "literature"]
    theory_hits = sum(text_lower.count(k) for k in theory_keywords)
    application_score = min(
        QUALIFI_RUBRIC["Application of Theory and Literature"],
        10 + theory_hits * 3,
    )

    # Referencing – look for urls and parentheses as crude citations
    ref_hits = text_lower.count("http") + text_lower.count("www")
    ref_hits += assignment_text.count("(")
    referencing_score = min(
        QUALIFI_RUBRIC["Referencing"],
        10 + ref_hits * 2,
    )

    # Presentation – based on average words per line
    line_count = assignment_text.count("\n") + 1
    avg_words_per_line = word_count / max(line_count, 1)
    if avg_words_per_line < 8:
        presentation_score = 20
    elif avg_words_per_line < 15:
        presentation_score = 30
    else:
        presentation_score = QUALIFI_RUBRIC["Presentation/Writing Skills"]

    mark_breakdown = {
        "Content": {"score": content_score},
        "Application of Theory and Literature": {"score": application_score},
        "Knowledge and Understanding": {"score": knowledge_score},
        "Presentation/Writing Skills": {"score": presentation_score},
        "Referencing": {"score": referencing_score},
    }

    # Add justifications per area using templates
    for area, max_score in QUALIFI_RUBRIC.items():
        s = mark_breakdown[area]["score"]
        mark_breakdown[area]["justification"] = feedback_for_score(s, max_score, area)

    total_score = sum(mark_breakdown[k]["score"] for k in mark_breakdown)
    percentage = round(total_score / 220 * 100, 1)
    grade = calculate_grade(percentage)

    # Unit code detection
    unit_code = "AID 401"
    for code in UNIT_CRITERIA.keys():
        if code in unit_title:
            unit_code = code
            break

    # Criteria feedback using simple keyword match
    criteria_feedback = []
    for crit in UNIT_CRITERIA.get(unit_code, []):
        key_words = [w.lower().strip(",.") for w in crit.split()[:4]]
        achieved = all(kw in text_lower for kw in key_words)
        criteria_feedback.append(
            {
                "criterion": crit,
                "achieved": achieved,
                "evidence": "The learner's response includes several key phrases related to this criterion."
                if achieved
                else "Key phrases for this criterion are limited or not clearly evidenced in the response.",
                "improvement": "Add a focused paragraph that explicitly addresses this criterion with concrete examples.",
            }
        )

    # Strengths and improvements lists (by area)
    strengths = [
        area
        for area, data in mark_breakdown.items()
        if data["score"] >= 0.7 * QUALIFI_RUBRIC[area]
    ]
    improvements = [
        area
        for area, data in mark_breakdown.items()
        if data["score"] < 0.7 * QUALIFI_RUBRIC[area]
    ]

    if strengths:
        strengths_text = [
            f"{area}: {mark_breakdown[area]['justification']}" for area in strengths
        ]
    else:
        strengths_text = ["No particular strengths were identified; all areas require further development."]

    if improvements:
        improvements_text = [
            f"{area}: {mark_breakdown[area]['justification']}" for area in improvements
        ]
    else:
        improvements_text = ["All areas are currently performing at a strong level."]

    overall_feedback = (
        "This evaluation has been generated using an automated rubric‑based scoring tool. "
        "Scores reflect length, basic use of theoretical language, structure and evidence of referencing. "
        "Learners and assessors should use the section comments to guide targeted improvements."
    )

    evaluation = {
        "unit_code": unit_code,
        "mark_breakdown": mark_breakdown,
        "total_score": total_score,
        "percentage": percentage,
        "grade": grade,
        "criteria_feedback": criteria_feedback,
        "overall_feedback": overall_feedback,
        "strengths": strengths_text,
        "improvements": improvements_text,
        "evaluation": {},
    }
    return evaluation


def generate_pdf(student_name: str, student_id: str, unit_title: str, evaluation: dict) -> str:
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "QUALIFI LEVEL 4 DIPLOMA IN AI - EVALUATION REPORT", 0, 1, "C")
    pdf.ln(5)

    # Student info
    pdf.set_font("Arial", "", 11)
    pdf.cell(100, 7, f"Student Name: {student_name}", 0, 0)
    pdf.cell(0, 7, f"Student ID: {student_id}", 0, 1)
    pdf.cell(0, 7, f"Unit: {unit_title}", 0, 1)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

    # Overall grade band
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(220, 235, 255)
    grade = evaluation.get("grade", "N/A")
    score = evaluation.get("total_score", 0)
    perc = evaluation.get("percentage", 0)
    pdf.cell(
        0,
        12,
        f"FINAL GRADE: {grade} | SCORE: {score}/220 ({perc}%)",
        1,
        1,
        "C",
        True,
    )
    pdf.ln(5)

    # Mark scheme breakdown
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
        pdf.multi_cell(0, 5, data.get("justification", ""))
        pdf.ln(2)

    # Criteria feedback
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

    # Overall feedback
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. OVERALL EVALUATION", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, evaluation.get("overall_feedback", "N/A"))

    # Strengths and improvements
    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Key Strengths:", 0, 1)
    pdf.set_font("Arial", "", 10)
    for s in evaluation.get("strengths", []):
        pdf.multi_cell(0, 5, f"- {s}")
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Areas for Improvement:", 0, 1)
    pdf.set_font("Arial", "", 10)
    for i in evaluation.get("improvements", []):
        pdf.multi_cell(0, 5, f"- {i}")

    filename = f"tmp/qualifi-pdfs/{student_id}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf.output(filename)
    return filename


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

        evaluation = evaluate_locally(assignment_text, rubric_text, unit_title)
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
