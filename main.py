import os
import re
import json
import shutil
import spacy
from docx import Document
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pdfminer.high_level import extract_text
import google.generativeai as genai
import schemas
import service
import models

# --- Database URL ---
SQLALCHEMY_DATABASE_URL = (
    "postgresql://ryan_ai_aviation_user:riBVsmNCckogNtl9GAGKMe3dNXuKrizD"
    "@dpg-d2522j49c44c73b4f3e0-a.oregon-postgres.render.com:5432/ryan_ai_aviation"
    "?sslmode=require"
)

# --- SQLAlchemy setup ---
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=5, pool_timeout=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()
app.mount("/resumes", StaticFiles(directory="resumes"), name="resumes")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
genai.configure(api_key="AIzaSyCj6Tp-9kCDI3LoUiL6g9iMxuLOksFf-qE")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Helper Functions ---
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""


def extract_with_gemini(resume_text):
    prompt = f"""
    Extract the following from the resume:
    - Full Name
    - Email
    - Phone Number
    - Qualification
    - Years of Experience (as a number)
    - Career Objective
    - List of Technical Skills (comma-separated)
    - Location

    Return JSON in this format:
    {{
        "name": "...",
        "email": "...",
        "phone": "...",
        "qualification": "...",
        "experience": 3,
        "objective": "...",
        "skills": "...",
        "location": "..."
    }}

    Resume:
    \"\"\"{resume_text}\"\"\""""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Error parsing JSON:", e)
        return {}


def extract_job_keywords(description):
    prompt = f"""
    From the job description, extract:
    - Skills: list, comma separated
    - Experience: minimum years or range (e.g., "3-5", "2")

    Return JSON:
    {{
        "skills": ["skill1", "skill2", ...],
        "experience": "3-5"
    }}

    Job Description:
    \"\"\"{description}\"\"\""""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", response.text.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except:
        return {"skills": [], "experience": ""}


def build_user_keywords(gpt_data):
    return {
        "skills": [s.strip().lower() for s in gpt_data.get("skills", "").split(",") if s.strip()],
        "experience": int(gpt_data.get("experience", 0))
        if str(gpt_data.get("experience", "0")).isdigit()
        else 0,
    }


def build_jobs_keywords(db: Session):
    jobs_data = db.execute(
        text('SELECT * FROM "Job" WHERE status = \'ACTIVE\'')
    ).mappings().all()

    jobs_with_keywords = []
    for job in jobs_data:
        job_keywords = extract_job_keywords(job["description"])
        jobs_with_keywords.append({
            **job,
            "skills_required": [s.lower() for s in job_keywords["skills"]],
            "experience_required": job_keywords["experience"]
        })
    return jobs_with_keywords


def compare_keywords(user_keywords, job_keywords):
    matches = []
    for job in job_keywords:
        skill_match = any(
            user_skill in job_skill or job_skill in user_skill
            for user_skill in user_keywords["skills"]
            for job_skill in job["skills_required"]
        )
        exp_match = False
        try:
            job_exp = job["experience_required"]
            if "-" in job_exp:
                low, high = job_exp.split("-")
                exp_match = int(low.strip()) <= user_keywords["experience"] <= int(
                    high.strip())
            elif job_exp.strip().isdigit():
                exp_match = user_keywords["experience"] >= int(job_exp.strip())
            else:
                exp_match = True
        except:
            exp_match = True

        if skill_match and exp_match:
            matches.append(job)
    return matches


# --- Routes ---
@app.get("/jobs/api", response_model=list[schemas.job])
def get_jobs_api(db: Session = Depends(get_db)):
    return service.get_job(db)


@app.get("/jobs")
async def index(request: Request, db: Session = Depends(get_db)):
    jobs = service.get_job(db)
    return templates.TemplateResponse("index.html", {"request": request, "all_jobs": jobs})


@app.post("/upload")
async def upload_resume(request: Request, resume: UploadFile = File(...), db: Session = Depends(get_db)):
    if not resume.filename:
        return JSONResponse("No file selected.", status_code=400)

    file_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    try:
        ext = os.path.splitext(resume.filename)[-1].lower()
        if ext == ".pdf":
            resume_text = extract_text(file_path)
        elif ext == ".docx":
            resume_text = extract_text_from_docx(file_path)
        else:
            return JSONResponse("Unsupported file format", status_code=400)

        # Extract structured resume data using Gemini
        gpt_data = extract_with_gemini(resume_text)

    except Exception as e:
        gpt_data = {}
        print("AI extraction failed:", e)

    # Build user keywords
    user_keywords = build_user_keywords(gpt_data)

    # Build job keywords from DB
    jobs_with_keywords = build_jobs_keywords(db)

    # Compare and find matches
    recommended_jobs = compare_keywords(user_keywords, jobs_with_keywords)

    # Pass data to template
    return {
        "recommended_jobs": recommended_jobs
    }
