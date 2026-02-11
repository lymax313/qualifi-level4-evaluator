from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from fpdf import FPDF
import uvicorn
import json
import re
import os
from datetime import datetime
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qualifi Level 4 Evaluator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
os.makedirs("tmp/qualifi-pdfs", exist_ok=True)

class GradeBand(str, Enum):
    DISTINCTION = "Distinction"
    MERIT = "Merit"
    PASS = "Pass"
    REFER = "Refer"

class CriterionResult(BaseModel):
    score: int = Field(..., ge=0)
    max_score: int
    comment: str

class Flags(BaseModel):
    off_topic: bool = False
    not_level_4: bool = False
    empty_or_invalid_submission: bool = False
    possible_academic_misconduct: bool = False

class EvaluationRequest(BaseModel):
    unit_code: str
    rubric_text: str
    assignment_text: str
    learner_name: Optional[str] = ""
    learner_id: Optional[str] = ""

class EvaluationResponse(BaseModel):
    unit_code: str
    learner_name: Optional[str]
    learner_id: Optional[str]
    criteria: Dict[str, CriterionResult]
    total_score: int = Field(..., ge=0, le=100)
    grade_band: GradeBand
    overall_justification: str
    flags: Flags

RUBRIC_DESCRIPTORS = {
    "CONTENT": {"max_score": 30, "bands": {"Distinction": (24, 30), "Merit": (18, 23), "Pass": (12, 17), "Refer": (0, 11)}},
    "THEORY": {"max_score": 25, "bands": {"Distinction": (20, 25), "Merit": (15, 19), "Pass": (10, 14), "Refer": (0, 9)}},
    "UNDERSTANDING": {"max_score": 25, "bands": {"Distinction": (20, 25), "Merit": (15, 19), "Pass": (10, 14), "Refer": (0, 9)}},
    "PRESENTATION": {"max_score": 20, "bands": {"Distinction": (16, 20), "Merit": (12, 15), "Pass": (8, 11), "Refer": (0, 7)}}
}

def calculate_grade_band(total_score: int) -> GradeBand:
    if total_score >= 70: return GradeBand.DISTINCTION
    elif total_score >= 60: return GradeBand.MERIT
    elif total_score >= 40: return GradeBand.PASS
    else: return GradeBand.REFER

def check_empty_or_invalid(text: str) -> bool:
    return not text or len(text.strip()) < 50 or len(text.split()) < 20

def check_off_topic(text: str, unit_code: str) -> bool:
    text_lower = text.lower()
    keywords = ["analysis", "discuss", "evaluate", "theory", "concept", "research", "evidence", "argument", "conclusion", "introduction"]
    return sum(1 for kw in keywords if kw in text_lower) < 2 and len(text) < 200

def check_academic_misconduct(text: str) -> bool:
    phrases = ["according to wikipedia", "is defined as", "refers to the", "is a term used"]
    return sum(1 for p in phrases if p in text.lower()) >= 3

def check_level_4_standard(text: str) -> bool:
    basic = ["i think", "in my opinion", "very good", "really nice"]
    advanced = ["methodology", "paradigm", "meta-analysis"]
    return not (sum(1 for b in basic if b in text.lower()) >= 5 and len(text.split()) < 300)

def evaluate_criterion(criterion: str, text: str) -> CriterionResult:
    config = RUBRIC_DESCRIPTORS[criterion]
    max_score = config["max_score"]
    text_lower = text.lower()
    word_count = len(text.split())
    
    keywords_map = {
        "CONTENT": (["evaluate", "analyze", "critique"], ["synthesize", "integrate"], ["argue", "suggest", "propose"]),
        "THEORY": (["theory", "framework", "model"], ["according to", "research", "literature"], []),
        "UNDERSTANDING": (["understand", "demonstrate", "explain"], ["comprehensive", "detailed"], []),
        "PRESENTATION": (["logical", "coherent", "organized"], ["clear", "polished"], [])
    }
    
    if criterion in keywords_map:
        kw1, kw2, kw3 = keywords_map[criterion]
        count1 = sum(1 for k in kw1 if k in text_lower)
        count2 = sum(1 for k in kw2 if k in text_lower)
        
        if count1 >= 4 and count2 >= 3: score = max_score - 4
        elif count1 >= 3 and count2 >= 2: score = int(max_score * 0.75)
        elif count1 >= 2: score = int(max_score * 0.6)
        elif count1 >= 1: score = int(max_score * 0.45)
        else: score = int(max_score * 0.25) if word_count > 200 else 0
    else:
        score = int(max_score * 0.5)
    
    comment = f"Score {score}/{max_score}. Assignment demonstrates {'excellent' if score > max_score*0.8 else 'good' if score > max_score*0.6 else 'adequate' if score > max_score*0.4 else 'limited'} performance in {criterion.lower()}. Areas for development: increase depth and breadth of analysis."
    return CriterionResult(score=score, max_score=max_score, comment=comment)

def evaluate_assignment(unit_code: str, rubric_text: str, assignment_text: str, learner_name: str = "", learner_id: str = "") -> EvaluationResponse:
    flags = Flags()
    if check_empty_or_invalid(assignment_text):
        flags.empty_or_invalid_submission = True
    if check_off_topic(assignment_text, unit_code):
        flags.off_topic = True
    if not check_level_4_standard(assignment_text):
        flags.not_level_4 = True
    if check_academic_misconduct(assignment_text):
        flags.possible_academic_misconduct = True
    
    criteria_results = {}
    total_score = 0
    for criterion in ["CONTENT", "THEORY", "UNDERSTANDING", "PRESENTATION"]:
        result = evaluate_criterion(criterion, assignment_text)
        criteria_results[criterion] = result
        total_score += result.score
    
    grade_band = calculate_grade_band(total_score)
    justification = f"Total score: {total_score}/100 ({grade_band.value}). " + ("Content appears off-topic. " if flags.off_topic else "") + ("Work below Level 4 standard. " if flags.not_level_4 else ") + ("Possible academic misconduct detected. " if flags.possible_academic_misconduct else "")
    
    return EvaluationResponse(unit_code=unit_code, learner_name=learner_name, learner_id=learner_id, criteria=criteria_results, total_score=total_score, grade_band=grade_band, overall_justification=justification, flags=flags)

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(37, 99, 235)
        self.cell(0, 10, "Qualifi Level 4 - Evaluation Report", 0, 1, "C")
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_pdf(result: EvaluationResponse) -> str:
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Student Information", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(40, 6, "Unit Code", 0)
    pdf.cell(0, 6, result.unit_code, 0, 1)
    if result.learner_name:
        pdf.cell(40, 6, "Learner Name", 0)
        pdf.cell(0, 6, result.learner_name, 0, 1)
    if result.learner_id:
        pdf.cell(40, 6, "Learner ID", 0)
        pdf.cell(0, 6, result.learner_id, 0, 1)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Total Score: {result.total_score}/100 - {result.grade_band.value}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Criteria Scores", 0, 1)
    pdf.set_font("Arial", "", 10)
    for criterion, data in result.criteria.items():
        pdf.cell(80, 6, f"{criterion}: {data.score}/{data.max_score}", 0, 1)
    
    filename = f"tmp/qualifi-pdfs/evaluation_{result.unit_code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        result = evaluate_assignment(request.unit_code, request.rubric_text, request.assignment_text, request.learner_name or "", request.learner_id or "")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-file")
async def evaluate_file(unit_code: str = Form(...), rubric_text: str = Form(...), assignment_file: UploadFile = File(...), learner_name: str = Form(default=""), learner_id: str = Form(default="")):
    try:
        content = await assignment_file.read()
        try:
            assignment_text = content.decode('utf-8')
        except:
            assignment_text = content.decode('latin-1')
        result = evaluate_assignment(unit_code, rubric_text, assignment_text, learner_name, learner_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/download-pdf")
async def download_pdf(request: Request):
    try:
        data = await request.json()
        result = EvaluationResponse(**data)
        pdf_path = generate_pdf(result)
        return FileResponse(pdf_path, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return open("index.html", "r").read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
