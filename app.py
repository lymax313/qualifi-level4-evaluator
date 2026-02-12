from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import google.generativeai as genai
import uvicorn
import os
import json
import re
from datetime import datetime
from fpdf import FPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Qualifi Level 4 Evaluator", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

# Qualifi Marking Rubric (Total: 220 points)
QUALIFI_RUBRIC = {
    "Content": 50,
    "Application of Theory and Literature": 40,
    "Knowledge and Understanding": 50,
    "Presentation/Writing Skills": 40,
    "Referencing": 40
}

GRADE_BANDS = [
    {"name": "Distinction", "min": 70, "max": 100},
    {"name": "Merit", "min": 60, "max": 69},
    {"name": "Pass", "min": 40, "max": 59},
    {"name": "FAIL", "min": 0, "max": 39}
]

def parse_rubric_file(content: bytes, filename: str) -> str:
    """Parse rubric from uploaded file - extract summative assessment criteria only."""
    try:
        rubric_text = content.decode('utf-8', errors='ignore')
        
        # Extract only SUMMATIVE assessment sections (ignore FORMATIVE)
        summative_pattern = r'SUMMATIVE\s+TASK.*?(?=FORMATIVE|$)'
        summative_match = re.search(summative_pattern, rubric_text, re.DOTALL | re.IGNORECASE)
        
        if summative_match:
            summative_text = summative_match.group()
            logger.info(f"Extracted summative assessment criteria: {len(summative_text)} chars")
            return summative_text
        
        # If no explicit SUMMATIVE section, look for Learning Outcomes and Assessment Criteria
        if "Learning Outcomes" in rubric_text and "Assessment Criteria" in rubric_text:
            logger.info(f"Using full rubric with Learning Outcomes: {len(rubric_text)} chars")
            return rubric_text
        
        if len(rubric_text.strip()) > 100:
            return rubric_text
        
        return None
    except Exception as e:
        logger.error(f"Rubric parsing error: {str(e)}")
        return None

async def evaluate_with_gemini(assignment_text: str, rubric_text: str, unit_title: str) -> dict:
    """Evaluate assignment using Qualifi summative assessment criteria."""
    try:
        if not GEMINI_API_KEY:
            return generate_fallback_evaluation()
        
        if not rubric_text or len(rubric_text.strip()) < 100:
            return generate_fallback_evaluation()
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a Qualifi Level 4 academic evaluator for the unit: {unit_title}

SUMMATIVE ASSESSMENT RUBRIC (Focus ONLY on summative criteria, ignore formative):
{rubric_text[:3000]}

QUALIFI MARKING RUBRIC (Total 220 points):
1. Content: 50 points
2. Application of Theory and Literature: 40 points
3. Knowledge and Understanding: 50 points
4. Presentation/Writing Skills: 40 points
5. Referencing: 40 points

ASSIGNMENT TO EVALUATE:
{assignment_text[:4000]}

Evaluate this assignment STRICTLY according to the SUMMATIVE assessment criteria in the rubric.

For EACH Learning Outcome in the summative section, assess if the student has achieved it (Y/N) and provide evidence.

Then provide scores for each of the 5 Qualifi marking criteria.

Return ONLY valid JSON in this EXACT format:
{{
  "learning_outcomes": [
    {{"outcome": "LO description", "achieved": "Y or N", "evidence": "specific evidence from assignment"}}
  ],
  "rubric_scores": {{
    "Content": {{"score": 0-50, "feedback": ""}},
    "Application of Theory and Literature": {{"score": 0-40, "feedback": ""}},
    "Knowledge and Understanding": {{"score": 0-50, "feedback": ""}},
    "Presentation/Writing Skills": {{"score": 0-40, "feedback": ""}},
    "Referencing": {{"score": 0-40, "feedback": ""}}
  }},
  "overall_feedback": "",
  "strengths": [],
  "improvements": []
}}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            
            # Calculate total score
            total_score = 0
            if "rubric_scores" in evaluation:
                for criterion, max_score in QUALIFI_RUBRIC.items():
                    if criterion in evaluation["rubric_scores"]:
                        score = evaluation["rubric_scores"][criterion].get("score", 0)
                        total_score += score
            
            evaluation["total_score"] = total_score
            evaluation["percentage"] = round((total_score / 220) * 100, 1)
            evaluation["grade"] = calculate_grade(evaluation["percentage"])
            
            return evaluation
        else:
            return generate_fallback_evaluation()
            
    except Exception as e:
        logger.error(f"Gemini evaluation error: {str(e)}")
        return generate_fallback_evaluation()

def generate_fallback_evaluation() -> dict:
    """Fallback evaluation when Gemini fails."""
    return {
        "learning_outcomes": [
            {"outcome": "Unable to evaluate", "achieved": "N", "evidence": "Evaluation system error"}
        ],
        "rubric_scores": {
            "Content": {"score": 25, "feedback": "Unable to fully evaluate. Please review manually."},
            "Application of Theory and Literature": {"score": 20, "feedback": "Unable to fully evaluate."},
            "Knowledge and Understanding": {"score": 25, "feedback": "Unable to fully evaluate."},
            "Presentation/Writing Skills": {"score": 20, "feedback": "Unable to fully evaluate."},
            "Referencing": {"score": 20, "feedback": "Unable to fully evaluate."}
        },
        "total_score": 110,
        "percentage": 50.0,
        "grade": "Pass",
        "overall_feedback": "System error occurred. Manual review required.",
        "strengths": [],
        "improvements": []
    }

def calculate_grade(percentage: float) -> str:
    """Calculate grade band based on percentage."""
    for band in GRADE_BANDS:
        if band["min"] <= percentage <= band["max"]:
            return band["name"]
    return "Ungraded"

def generate_pdf(student_name: str, student_id: str, unit_title: str, evaluation: dict) -> str:
    """Generate PDF report with Qualifi formatting."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Qualifi Level 4 Assignment Evaluation", 0, 1, "C")
    pdf.ln(5)
    
    # Student Info
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Student Name: {student_name}", 0, 1)
    pdf.cell(0, 7, f"Student ID: {student_id}", 0, 1)
    pdf.cell(0, 7, f"Unit Title: {unit_title}", 0, 1)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)
    
    # Overall Grade
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(220, 220, 220)
    grade = evaluation.get("grade", "Ungraded")
    total = evaluation.get("total_score", 0)
    percentage = evaluation.get("percentage", 0)
    pdf.cell(0, 10, f"Final Grade: {grade} ({total}/220 = {percentage}%)", 0, 1, "C", True)
    pdf.ln(5)
    
    # Learning Outcomes
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Learning Outcomes Assessment:", 0, 1)
    pdf.ln(2)
    
    for lo in evaluation.get("learning_outcomes", []):
        pdf.set_font("Arial", "B", 10)
            achieved = "[PASS] ACHIEVED" if lo.get("achieved") == "Y" else "[FAIL] NOT ACHIEVED"                    pdf.cell(0, 6, f"{achieved}: {lo.get('outcome', 'N/A')[:80]}", 0, 1)
                    pdf.set_font("Arial", "", 9)
                    pdf.multi_cell(0, 5, f"Evidence: {lo.get('evidence', 'N/A')[:150]}")
                    
    
    # Rubric Scores
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Qualifi Marking Rubric Scores:", 0, 1)
    pdf.ln(2)
    
    for criterion, max_score in QUALIFI_RUBRIC.items():
        if criterion in evaluation.get("rubric_scores", {}):
            data = evaluation["rubric_scores"][criterion]
            score = data.get("score", 0)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, f"{criterion}: {score}/{max_score}", 0, 1)
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5, data.get("feedback", "N/A")[:200])
            pdf.ln(2)
    
    # Overall Feedback
    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Overall Feedback:", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, evaluation.get("overall_feedback", "N/A"))
    
    filename = f"tmp/qualifi-pdfs/{student_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/evaluate")
async def evaluate_assignment(
    student_name: str = Form(...),
    student_id: str = Form(...),
    unit_title: str = Form(...),
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...)
):
    try:
        # Read rubric file (extract summative criteria only)
        rubric_content = await rubric.read()
        rubric_text = parse_rubric_file(rubric_content, rubric.filename)
        
        if not rubric_text:
            raise HTTPException(status_code=400, detail="Unable to parse rubric file. Please upload a valid text-based rubric.")
        
        # Read assignment file
        assignment_content = await assignment.read()
        assignment_text = assignment_content.decode('utf-8', errors='ignore')
        
        if len(assignment_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Assignment text too short")
        
        # Evaluate using Qualifi rubric
        evaluation = await evaluate_with_gemini(assignment_text, rubric_text, unit_title)
        
        # Generate PDF report
        pdf_path = generate_pdf(student_name, student_id, unit_title, evaluation)
        
        return JSONResponse({
            "success": True,
            "student_name": student_name,
            "student_id": student_id,
            "unit_title": unit_title,
            "evaluation": evaluation,
            "pdf_url": f"/download/{os.path.basename(pdf_path)}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = f"tmp/qualifi-pdfs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
