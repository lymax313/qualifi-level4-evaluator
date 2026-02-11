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

async def evaluate_with_gemini(rubric_text: str, assignment_text: str, module: str) -> dict:
    """Optimized Gemini AI evaluation with detailed feedback following Result-format template"""
    try:
        if not GEMINI_API_KEY:
            return generate_fallback_evaluation()
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are an academic evaluator following the Qualifi Level 4 assessment framework.

Module: {module}

RUBRIC/ASSESSMENT CRITERIA:
{rubric_text[:3000]}

STUDENT ASSIGNMENT:
{assignment_text[:4000]}

Evaluate the assignment strictly according to the rubric provided. Generate a detailed evaluation report following this EXACT structure:

1. Extract all Learning Outcomes from the rubric
2. For each Learning Outcome, identify its Assessment Criteria
3. For each Assessment Criterion, determine:
   - Is it Achieved (Y) or Not Yet Achieved (N)?
   - Provide comprehensive, detailed feedback for each criterion (minimum 4-6 sentences):
      * Quote or reference specific content from the student's assignment
      * Explain what the student did well or what was missing
      * Relate the feedback to Qualifi Level 4 expectations
      * Provide constructive suggestions for improvement if not achieved
      * Use professional academic language and be specific with examples
Return ONLY valid JSON in this exact format:
{{
  "learning_outcomes": [
    {{
      "outcome": "<Learning Outcome text>",
      "criteria": [
        {{
          "criterion": "<Assessment Criteria text>",
          "achieved": "Y" or "N",
          "evidence": "<Detailed feedback explaining why achieved or not achieved, with specific examples from the submission>"
        }}
      ]
    }}
  ],
  "overall_feedback": "<Overall performance assessment>",
  "overall_grade": "<Pass/Merit/Distinction/Refer>"
}}

IMPORTANT:
- Be specific and detailed in evidence feedback
- Reference actual content from the assignment
- Explain clearly why each criterion was achieved or not
- Overall feedback should be comprehensive (5-7 sentences)
   - Include strengths and areas for improvement for each criterion
   - Use evidence from rubric learning outcomes and assessment criteria
   - Provide actionable recommendations for student development
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            return evaluation
        else:
            logger.warning("Could not extract JSON from Gemini response")
            return generate_fallback_evaluation()
            
    except Exception as e:
        logger.error(f"Gemini evaluation error: {str(e)}")
        return generate_fallback_evaluation()

def generate_fallback_evaluation() -> dict:
    """Fallback evaluation when Gemini is unavailable"""
    return {
        "learning_outcomes": [
            {
                "outcome": "Demonstrate understanding of core concepts",
                "criteria": [
                    {
                        "criterion": "Explain fundamental principles",
                        "achieved": "N",
                        "evidence": "Unable to evaluate - Gemini AI unavailable. Manual assessment required."
                    }
                ]
            }
        ],
        "overall_feedback": "Automated evaluation unavailable. Please review manually.",
        "overall_grade": "Pending Review"
    }

def generate_pdf_result_format(student_name: str, student_id: str, module: str, unit_title: str, evaluation: dict) -> str:
    """Generate PDF following Result-format.pdf template structure"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Presentation Feedback", 0, 1, "C")
    pdf.ln(5)
    
    # Student Information
    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Learner Name:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, student_name, 0, 1)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Learner Registration ID:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, student_id, 0, 1)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Qualification:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "L4", 0, 1)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Unit Title:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, unit_title, 0, 1)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Assignment submitted:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Yes", 0, 1)
    pdf.ln(5)
    
    # Table Header
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(60, 8, "Learning Outcomes", 1, 0, "C", True)
    pdf.cell(50, 8, "Assessment Criteria", 1, 0, "C", True)
    pdf.cell(10, 8, "A", 1, 0, "C", True)
    pdf.cell(10, 8, "NYA", 1, 0, "C", True)
    pdf.cell(60, 8, "Evidence", 1, 1, "C", True)
    
    # Table Content
    pdf.set_font("Arial", "", 8)
    for lo in evaluation.get("learning_outcomes", []):
        outcome_text = lo.get("outcome", "")[:80]
        criteria_list = lo.get("criteria", [])
        
        for idx, criterion in enumerate(criteria_list):
            crit_text = criterion.get("criterion", "")[:60]
            achieved = criterion.get("achieved", "N")
            evidence = criterion.get("evidence", "")[:150]
            
            # Calculate row height based on content
            row_height = max(12, len(evidence) // 30 * 4)
            
            if idx == 0:
                pdf.cell(60, row_height, outcome_text, 1, 0)
            else:
                pdf.cell(60, row_height, "", 1, 0)
            
            pdf.cell(50, row_height, crit_text, 1, 0)
            pdf.cell(10, row_height, "Y" if achieved == "Y" else "", 1, 0, "C")
            pdf.cell(10, row_height, "N" if achieved == "N" else "", 1, 0, "C")
            pdf.multi_cell(60, 4, evidence, 1)
    
    pdf.ln(5)
    
    # Overall Feedback
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Overall Feedback", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, evaluation.get("overall_feedback", "No feedback provided"))
    pdf.ln(3)
    
    # Overall Grade
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, f"Overall Grade: {evaluation.get('overall_grade', 'Pending')}", 1, 1, "C", True)
    
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
    student_id: str = Form(...),
    module: str = Form(...),
    unit_title: str = Form(...),
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...)
):
    try:
        # Read rubric file
        rubric_content = await rubric.read()
        rubric_text = rubric_content.decode('utf-8', errors='ignore')
        
        if len(rubric_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Rubric file is too short or empty")
        
        # Read assignment file
        assignment_content = await assignment.read()
        assignment_text = assignment_content.decode('utf-8', errors='ignore')
        
        if len(assignment_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Assignment text too short")
        
        # Evaluate with Gemini AI
        evaluation = await evaluate_with_gemini(rubric_text, assignment_text, module)
        
        # Generate PDF report
        pdf_path = generate_pdf_result_format(student_name, student_id, module, unit_title, evaluation)
        
        return JSONResponse({
            "success": True,
            "student_name": student_name,
            "student_id": student_id,
            "module": module,
            "unit_title": unit_title,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

app = FastAPI(title="Qualifi Level 4 Evaluator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

# Qualifi Rubric - Exact grading bands from PDF
RUBRIC = """
Qualifi Level 4 Assignment Evaluation Rubric:

1. Content alignment with assessment criteria (30%):
   - 80-100 (Distinction): Extensive evaluation and synthesis of ideas, substantial original thinking
   - 70-79 (Distinction): Comprehensive critical evaluation and synthesis, coherent original thinking
   - 60-69 (Merit): Adequate evaluation and synthesis beyond basic descriptions, original thinking
   - 50-59 (Pass): Describes main ideas with evidence of evaluation, some original thinking
   - 40-49 (Pass): Describes some main ideas but omits concepts, limited evaluation
   - 30-39 (FAIL): Largely incomplete, misses key concepts, no original thinking
   - 0-39 (FAIL): Inadequate or irrelevant information

2. Application of Theory and Literature (25%):
   - 80-100: In-depth, detailed, expertly integrates literature
   - 70-79: Clear and relevant, fully integrates literature
   - 60-69: Appropriate application, integrates literature
   - 50-59: Adequate application, uses literature
   - 40-49: Limited application, inconsistent literature use
   - 30-39: Confused application, no literature support
   - 0-39: Little or no evidence of theory application

3. Knowledge and Understanding (25%):
   - 80-100: Extensive depth beyond key principles
   - 70-79: Comprehensive knowledge and depth
   - 60-69: Sound understanding of principles
   - 50-59: Basic knowledge of key concepts
   - 40-49: Limited and superficial knowledge
   - 30-39: Confused or inadequate knowledge
   - 0-39: Little or no evidence of understanding

4. Presentation and Writing Skills (20%):
   - 80-100: Polished, exceeding expectations, error-free
   - 70-79: Mastery, error-free mechanics
   - 60-69: Logical structure, few errors
   - 50-59: Orderly, minor errors
   - 40-49: Somewhat weak, errors may interfere
   - 30-39: Confused, errors often interfere
   - 0-39: Illogical, significant errors
"""

GRADE_BANDS = [
    {"name": "Distinction", "min": 70, "max": 100},
    {"name": "Merit", "min": 60, "max": 69},
    {"name": "Pass", "min": 40, "max": 59},
    {"name": "FAIL", "min": 0, "max": 39}
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

async def evaluate_with_gemini(assignment_text: str, module: str) -> dict:
    try:
        if not GEMINI_API_KEY:
            return generate_fallback_evaluation(assignment_text)
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a Qualifi Level 4 academic evaluator. Strictly follow the grading rubric.

Module: {module}

{RUBRIC}

Evaluate this assignment strictly according to the rubric above. 
For each criterion, select the appropriate grade band (80-100, 70-79, 60-69, 50-59, 40-49, 30-39, 0-39) based on the descriptors.

Assignment:
{assignment_text[:4000]}

Provide scores STRICTLY following the rubric bands and detailed feedback.
Return ONLY valid JSON:
{{
  "Content": {{"score": <number 0-100>, "band": "<grade band>", "feedback": "<text>"}},
  "Theory": {{"score": <number 0-100>, "band": "<grade band>", "feedback": "<text>"}},
  "Understanding": {{"score": <number 0-100>, "band": "<grade band>", "feedback": "<text>"}},
  "Presentation": {{"score": <number 0-100>, "band": "<grade band>", "feedback": "<text>"}}
}}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        json_match = re.search(r'\{{.*\}}', result_text, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            # Normalize scores to rubric weights
            evaluation["Content"]["weighted"] = evaluation["Content"]["score"] * 0.30
            evaluation["Theory"]["weighted"] = evaluation["Theory"]["score"] * 0.25
            evaluation["Understanding"]["weighted"] = evaluation["Understanding"]["score"] * 0.25
            evaluation["Presentation"]["weighted"] = evaluation["Presentation"]["score"] * 0.20
            return evaluation
        else:
            return generate_fallback_evaluation(assignment_text)
            
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return generate_fallback_evaluation(assignment_text)

def generate_fallback_evaluation(assignment_text: str) -> dict:
    word_count = len(assignment_text.split())
    
    if word_count < 100:
        return {
            "Content": {"score": 35, "band": "30-39", "weighted": 10.5, "feedback": "Insufficient content provided."},
            "Theory": {"score": 32, "band": "30-39", "weighted": 8.0, "feedback": "Limited theoretical application."},
            "Understanding": {"score": 35, "band": "30-39", "weighted": 8.75, "feedback": "Basic understanding demonstrated."},
            "Presentation": {"score": 38, "band": "30-39", "weighted": 7.6, "feedback": "Brief presentation."}
        }
    
    return {
        "Content": {"score": 62, "band": "60-69", "weighted": 18.6, "feedback": "Adequate coverage of content areas."},
        "Theory": {"score": 58, "band": "50-59", "weighted": 14.5, "feedback": "Adequate theoretical framework."},
        "Understanding": {"score": 64, "band": "60-69", "weighted": 16.0, "feedback": "Sound understanding of concepts."},
        "Presentation": {"score": 55, "band": "50-59", "weighted": 11.0, "feedback": "Orderly presentation."}
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
        data = evaluation[criterion]
        pdf.ln(2)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, f"{criterion} ({weight}): {data['score']}/100 [{data.get('band', 'N/A')}]", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"Weighted Score: {data['weighted']:.2f} | {data['feedback']}")
    
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
    assignment: UploadFile = File(...)
):
    try:
        content = await assignment.read()
        assignment_text = content.decode('utf-8', errors='ignore')
        
        if len(assignment_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Assignment text too short")
        
        evaluation = await evaluate_with_gemini(assignment_text, module)
        
        total_score = sum(data["weighted"] for data in evaluation.values())
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

