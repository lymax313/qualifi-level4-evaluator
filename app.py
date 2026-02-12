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

app = FastAPI(title="Qualifi Level 4 Evaluator", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

# AI Modules
MODULES = [
    "DAI401 - Introduction to Artificial Intelligence and Applications",
    "DAI402 - Mathematical Foundations for Machine Learning",
    "DAI403 - Data Science Using Python",
    "DAI404 - Big Data Management",
    "DAI405 - Introduction to Deep Learning",
    "DAI406 - Artificial Intelligence Ethics"
]

GRADE_BANDS = [
    {"name": "Distinction", "min": 70, "max": 100},
    {"name": "Merit", "min": 60, "max": 69},
    {"name": "Pass", "min": 40, "max": 59},
    {"name": "FAIL", "min": 0, "max": 39}
]

def parse_rubric_file(content: bytes, filename: str) -> str:
    """Parse rubric from uploaded file."""
    try:
        # Try decoding as text
        rubric_text = content.decode('utf-8', errors='ignore')
        if len(rubric_text.strip()) > 20:
            return rubric_text
        return None
    except Exception as e:
        logger.error(f"Rubric parsing error: {str(e)}")
        return None

async def evaluate_with_gemini(assignment_text: str, rubric_text: str, module: str) -> dict:
    """Evaluate assignment using uploaded rubric."""
    try:
        if not GEMINI_API_KEY:
            return generate_fallback_evaluation()
        
        if not rubric_text or len(rubric_text.strip()) < 20:
            return generate_fallback_evaluation()
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a Qualifi Level 4 academic evaluator.

Module: {module}

RUBRIC/ASSESSMENT CRITERIA:
{rubric_text[:2000]}

Evaluate this assignment STRICTLY according to the rubric criteria provided above.
For each criterion in the rubric, assess the student's work and provide:
1. Score (0-100)
2. Grade band (Pass/Merit/Distinction/FAIL)
3. Detailed feedback with evidence

Assignment:
{assignment_text[:4000]}

Return ONLY valid JSON in this exact format:
{{
    "Content": {{"score": <number>, "band": "<band>", "feedback": "<text>"}},
    "Theory": {{"score": <number>, "band": "<band>", "feedback": "<text>"}},
    "Understanding": {{"score": <number>, "band": "<band>", "feedback": "<text>"}},
    "Presentation": {{"score": <number>, "band": "<band>", "feedback": "<text>"}}
}}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            
            # Safely calculate weights
            for key in ["Content", "Theory", "Understanding", "Presentation"]:
                if key in evaluation and isinstance(evaluation[key], dict):
                    score = evaluation[key].get("score", 50)
                    weight = {"Content": 0.30, "Theory": 0.25, "Understanding": 0.25, "Presentation": 0.20}.get(key, 0.25)
                    evaluation[key]["weighted"] = score * weight
                else:
                    evaluation[key] = {"score": 50, "band": "50-59", "weighted": 12.5, "feedback": "No data"}
            
            return evaluation
        else:
            return generate_fallback_evaluation()
        
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return generate_fallback_evaluation()

def generate_fallback_evaluation() -> dict:
    """Fallback evaluation when Gemini fails."""
    return {
        "Content": {"score": 55, "band": "50-59", "weighted": 16.5, "feedback": "Unable to fully evaluate. Please review manually."},
        "Theory": {"score": 55, "band": "50-59", "weighted": 13.75, "feedback": "Unable to fully evaluate. Please review manually."},
        "Understanding": {"score": 55, "band": "50-59", "weighted": 13.75, "feedback": "Unable to fully evaluate. Please review manually."},
        "Presentation": {"score": 55, "band": "50-59", "weighted": 11.0, "feedback": "Unable to fully evaluate. Please review manually."}
    }

def calculate_grade(total_score: float) -> str:
    for band in GRADE_BANDS:
        if band["min"] <= total_score <= band["max"]:
            return band["name"]
    return "Ungraded"

def generate_pdf(student_name: str, student_id: str, module: str, evaluation: dict, total_score: float, grade: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Qualifi Level 4 Assignment Evaluation", 0, 1, "C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Student Name: {student_name}", 0, 1)
    pdf.cell(0, 7, f"Student ID: {student_id}", 0, 1)
    pdf.cell(0, 7, f"Module: {module}", 0, 1)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 10, f"Final Grade: {grade} ({total_score:.1f}/100)", 0, 1, "C", True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Evaluation:", 0, 1)
    pdf.ln(2)
    
    criteria_weights = {
        "Content": "30%",
        "Theory": "25%",
        "Understanding": "25%",
        "Presentation": "20%"
    }
    
    for criterion, weight in criteria_weights.items():
        if criterion in evaluation:
            data = evaluation[criterion]
            pdf.ln(2)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, f"{criterion} ({weight}): {data.get('score', 0)}/100 [{data.get('band', 'N/A')}]", 0, 1)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, f"Weighted Score: {data.get('weighted', 0):.2f} | {data.get('feedback', 'N/A')}")
    
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
    module: str = Form(...),
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...)
):
    try:
        # Read rubric file
        rubric_content = await rubric.read()
        rubric_text = parse_rubric_file(rubric_content, rubric.filename)
        
        if not rubric_text:
            raise HTTPException(status_code=400, detail="Unable to parse rubric file. Please upload a text-based file.")
        
        # Read assignment file
        assignment_content = await assignment.read()
        assignment_text = assignment_content.decode('utf-8', errors='ignore')
        
        if len(assignment_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Assignment text too short")
        
        # Evaluate using uploaded rubric
        evaluation = await evaluate_with_gemini(assignment_text, rubric_text, module)
        
        # Calculate total safely
        total_score = 0
        for key in ["Content", "Theory", "Understanding", "Presentation"]:
            if key in evaluation and isinstance(evaluation[key], dict):
                total_score += evaluation[key].get("weighted", 0)
        
        grade = calculate_grade(total_score)
        
        pdf_path = generate_pdf(student_name, student_id, module, evaluation, total_score, grade)
        
        return JSONResponse({
            "success": True,
            "student_name": student_name,
            "student_id": student_id,
            "module": module,
            "total_score": round(total_score, 2),
            "grade": grade,
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

@app.get("/api/modules")
async def get_modules():
    return {"modules": MODULES}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
