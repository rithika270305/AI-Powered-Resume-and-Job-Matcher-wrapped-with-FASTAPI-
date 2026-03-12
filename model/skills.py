import os
import json

BASE_DIR = os.path.dirname(__file__)
SKILLS_PATH = os.path.join(BASE_DIR, "skills.json")

with open(SKILLS_PATH) as f:
    SKILL_MAP = json.load(f)

# Build synonym → main skill map
SYNONYM_MAP = {}
for main_skill, synonyms in SKILL_MAP.items():
    SYNONYM_MAP[main_skill] = main_skill
    for syn in synonyms:
        SYNONYM_MAP[syn] = main_skill


def extract_skills(text: str):
    text = text.lower()
    found = set()

    for word in SYNONYM_MAP:
        if word in text:
            found.add(word)

    return list(found)


# ✅ MUST RETURN STRING OR NONE — NEVER LIST
def normalize_skill(skill: str):
    return SYNONYM_MAP.get(skill)