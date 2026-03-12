from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
import traceback

from model.matcher import match_resume_to_job

app = FastAPI()

class MatchRequest(BaseModel):
    resume_text: str
    job_text: str

class MatchResponse(BaseModel):
    overall_score: float
    semantic_score: float
    skill_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]


@app.post("/match")
def match_resume(req: MatchRequest = Body(...)):
    print("match HIT")

    try:
        result = match_resume_to_job(req.resume_text, req.job_text)
        print("RESULT:", result)
        return result

    except Exception as e:
        print("ERROR OCCURRED")
        traceback.print_exc()  
        return {"error": str(e)}