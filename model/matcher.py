import numpy as np
from .nlp_loader import get_embedding
from .similarity import cosine_similarity
from .preprocess import clean_text
from .skills import extract_skills, normalize_skill


def flatten_skills(skills_raw):
    """
    Ensures skills list is always flat.
    """
    flattened = []
    for item in skills_raw:
        if isinstance(item, list):
            flattened.extend(item)
        elif item:
            flattened.append(item)
    return flattened


def semantic_skill_match(job_skills, resume_skills, threshold=0.7):
    matched = []
    missing = []

    if not job_skills or not resume_skills:
        return matched, job_skills

    def safe_embedding(text):
        try:
            vec = get_embedding(text)
            if vec is None or len(vec) == 0:
                return np.zeros(384)  # MiniLM dimension
            return vec
        except Exception:
            return np.zeros(384)

    resume_vectors = {s: safe_embedding(s) for s in resume_skills}
    job_vectors = {s: safe_embedding(s) for s in job_skills}

    for job_skill, job_vec in job_vectors.items():
        max_similarity = 0.0

        for resume_skill, resume_vec in resume_vectors.items():
            score = cosine_similarity(job_vec, resume_vec)
            max_similarity = max(max_similarity, score)

        if max_similarity >= threshold:
            matched.append(job_skill)
        else:
            missing.append(job_skill)

    return matched, missing


def match_resume_to_job(resume_text: str, job_text: str):
    def safe_embedding(text):
        try:
            vec = get_embedding(text)
            if vec is None or len(vec) == 0:
                return np.zeros(384)
            return vec
        except Exception:
            return np.zeros(384)

    # 1️⃣ Clean text
    resume_clean = clean_text(resume_text or "")
    job_clean = clean_text(job_text or "")

    # 2️⃣ Full text semantic similarity
    resume_vector = safe_embedding(resume_clean)
    job_vector = safe_embedding(job_clean)
    semantic_score = cosine_similarity(resume_vector, job_vector)

    # 3️⃣ Extract skills
    resume_skills_raw = extract_skills(resume_clean) or []
    job_skills_raw = extract_skills(job_clean) or []

    # 4️⃣ Flatten
    resume_skills_flat = flatten_skills(resume_skills_raw)
    job_skills_flat = flatten_skills(job_skills_raw)

    # 5️⃣ Normalize (FINAL FIX)
    resume_skills = []
    for s in resume_skills_flat:
        norm = normalize_skill(s)
        if norm:
            resume_skills.append(norm)

    job_skills = []
    for s in job_skills_flat:
        norm = normalize_skill(s)
        if norm:
            job_skills.append(norm)

    resume_skills = list(set(resume_skills))
    job_skills = list(set(job_skills))

    # 6️⃣ Exact matches
    exact_matches = list(set(resume_skills) & set(job_skills))

    # 7️⃣ Semantic skill matching
    unmatched_job_skills = list(set(job_skills) - set(exact_matches))
    semantic_matches, missing_skills = semantic_skill_match(
        unmatched_job_skills, resume_skills
    )

    matched_skills = exact_matches + semantic_matches

    # 8️⃣ Skill match score
    skill_score = len(matched_skills) / len(job_skills) if job_skills else 0.0

    # 9️⃣ Overall score
    overall_score = (0.4 * semantic_score) + (0.6 * skill_score)

    return {
        "overall_score": round(overall_score * 100, 2),
        "semantic_score": round(semantic_score * 100, 2),
        "skill_match_score": round(skill_score * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }