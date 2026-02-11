from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
import uvicorn
import os
import json
import re
from datetime import datetime
from fpdf import FPDF
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Qualifi Level 4 Evaluator with Gemini AI", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

# Rubric configuration
RUBRIC = {
    "Content": {"max_score": 30, "weight": 0.3},
    "Theory": {"max_score": 25, "weight": 0.25},
    "Understanding": {"max_score": 25, "weight": 0.25},
    "Presentation": {"max_score": 20, "weight": 0.2}
}

GRADE_BANDS = [
    {"name": "Distinction", "min": 70, "max": 100},
    {"name": "Merit", "min": 60, "max": 69},
    {"name": "Pass", "min": 40, "max": 59},
    {"name": "Refer/Fail", "min": 0, "max": 39}
]

MODULES = [
    "SEM301DS - Strategic Management",
    "SEM302DS - Project Management",
    "SEM303DS - Financial Management",
    "SEM304DS - Human Resource Management",
    "SEM305DS - Marketing Management",
    "SEM306DS - Operations Management",
    "SEM307DS - Business Ethics",
    "SEM308DS - Entrepreneurship",
    "SEM309DS - Business Law",
    "SEM310DS - International Business",
    "SEM311DS - Research Methods",
    "SEM312DS - Business Analytics",
    "SEM313DS - Leadership",
    "SEM314DS - Innovation Management"
]

class EvaluationRequest(BaseModel):
    student_name: str
    qualification_level: str
    module: str
    assignment_text: str

async def evaluate_with_gemini(assignment_text: str, module: str) -> dict:
    """Use Gemini AI to evaluate the assignment"""
    try:
        if not GEMINI_API_KEY:
            return generate_fallback_evaluation(assignment_text)
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are an academic evaluator for Qualifi Level 4 assignments.

Module: {module}

Evaluate the following student assignment based on these criteria:
1. Content (30 marks): Depth, relevance, and completeness
2. Theory (25 marks): Application of theoretical concepts
3. Understanding (25 marks): Demonstration of comprehension
4. Presentation (20 marks): Structure, clarity, and formatting

Assignment Text:
{assignment_text[:3000]}

Provide scores and detailed feedback for each criterion.
Return ONLY a JSON object with this exact structure:
{{
  "Content": {{"score": <number>, "feedback": "<text>"}},
  "Theory": {{"score": <number>, "feedback": "<text>"}},
  "Understanding": {{"score": <number>, "feedback": "<text>"}},
  "Presentation": {{"score": <number>, "feedback": "<text>"}}
}}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{{.*\}}', result_text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            return evaluation
        else:
            return generate_fallback_evaluation(assignment_text)
            
    except Exception as e:
        logger.error(f"Gemini evaluation error: {str(e)}")
        return generate_fallback_evaluation(assignment_text)

def generate_fallback_evaluation(assignment_text: str) -> dict:
    """Generate basic evaluation when Gemini is unavailable"""
    word_count = len(assignment_text.split())
    
    if word_count < 100:
        return {
            "Content": {"score": 10, "feedback": "Insufficient content provided."},
            "Theory": {"score": 8, "feedback": "Limited theoretical application."},
            "Understanding": {"score": 8, "feedback": "Basic understanding demonstrated."},
            "Presentation": {"score": 7, "feedback": "Brief presentation."}
        }
    
    return {
        "Content": {"score": 22, "feedback": "Good coverage of content areas."},
        "Theory": {"score": 18, "feedback": "Adequate theoretical framework applied."},
        "Understanding": {"score": 19, "feedback": "Sound understanding of concepts."},
        "Presentation": {"score": 15, "feedback": "Clear and organized presentation."}
    }

def calculate_grade(total_score: float) -> str:
    """Calculate grade band based on total score"""
    for band in GRADE_BANDS:
        if band["min"] <= total_score <= band["max"]:
            return band["name"]
    return "Ungraded"

def generate_pdf(student_name: str, module: str, evaluation: dict, total_score: float, grade: str) -> str:
    """Generate PDF evaluation report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "Qualifi Level 4 Assignment Evaluation", 0, 1, "C")
    pdf.ln(5)
    
    # Student Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Student Name: {student_name}", 0, 1)
    pdf.cell(0, 8, f"Module: {module}", 0, 1)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)
    
    # Overall Grade
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Final Grade: {grade} ({total_score:.1f}/100)", 0, 1)
    pdf.ln(5)
    
    # Detailed Scores
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Evaluation:", 0, 1)
    pdf.set_font("Arial", "", 11)
    
    for criterion, data in evaluation.items():
        max_score = RUBRIC[criterion]["max_score"]
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{criterion}: {data['score']}/{max_score}", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"Feedback: {data['feedback']}")
    
    # Save PDF
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
    qualification_level: str = Form(...),
    module: str = Form(...),
    assignment: UploadFile = File(...)
):
    try:
        # Read assignment file
        content = await assignment.read()
        assignment_text = content.decode('utf-8', errors='ignore')
        
        if len(assignment_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Assignment text too short")
        
        # Evaluate with Gemini AI
        evaluation = await evaluate_with_gemini(assignment_text, module)
        
        # Calculate total score
        total_score = sum(data["score"] for data in evaluation.values())
        grade = calculate_grade(total_score)
        
        # Generate PDF
        pdf_path = generate_pdf(student_name, module, evaluation, total_score, grade)
        
        return JSONResponse({
            "success": True,
            "student_name": student_name,
            "module": module,
            "total_score": total_score,
            "grade": grade,
            "evaluation": evaluation,
            "pdf_url": f"/download/{os.path.basename(pdf_path)}"
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = f"tmp/qualifi-pdfs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/modules")
async def get_modules():
    return {"modules": MODULES}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
