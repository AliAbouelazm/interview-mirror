"""
Curated interview question bank. Six categories, written from the prompt set
recruiters and hiring managers actually use.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    category: str
    difficulty: str
    target_seconds: int


_QUESTIONS: list[Question] = [
    # Behavioural
    Question("b01", "Tell me about a time you disagreed with your manager. How did you handle it?", "behavioural", "medium", 90),
    Question("b02", "Walk me through the project you are most proud of. Why?", "behavioural", "easy", 120),
    Question("b03", "Describe a time you failed. What did you learn?", "behavioural", "medium", 90),
    Question("b04", "Tell me about a deadline you missed and what you did about it.", "behavioural", "medium", 90),
    Question("b05", "Describe a time you had to influence someone without authority.", "behavioural", "hard", 120),
    Question("b06", "Tell me about the most ambiguous problem you have solved.", "behavioural", "hard", 120),
    Question("b07", "Walk me through a tough decision you made with incomplete information.", "behavioural", "hard", 120),
    Question("b08", "Describe a time you received hard feedback. How did you react?", "behavioural", "medium", 90),

    # Leadership
    Question("l01", "How do you approach giving difficult feedback to a peer?", "leadership", "medium", 90),
    Question("l02", "Tell me about a time you had to lead without being the manager.", "leadership", "medium", 90),
    Question("l03", "How do you handle a teammate who is consistently underperforming?", "leadership", "hard", 120),
    Question("l04", "Describe how you would onboard a new engineer onto your team.", "leadership", "medium", 90),
    Question("l05", "What is your approach to setting priorities when the team is overcommitted?", "leadership", "hard", 120),

    # Technical thinking
    Question("t01", "Explain a technical concept you understand well to a non-technical person.", "technical", "easy", 120),
    Question("t02", "How would you decide between building a feature in-house versus buying it?", "technical", "medium", 120),
    Question("t03", "Walk me through how you would debug a sudden 5x latency spike in production.", "technical", "hard", 150),
    Question("t04", "Describe a system you have designed end to end. What were the tradeoffs?", "technical", "hard", 150),
    Question("t05", "How do you balance shipping fast against engineering quality?", "technical", "medium", 90),
    Question("t06", "What is the most interesting bug you have ever found?", "technical", "easy", 120),

    # Situational
    Question("s01", "If you were given a project with no clear scope, how would you start?", "situational", "medium", 90),
    Question("s02", "How would you handle a stakeholder who keeps changing requirements?", "situational", "medium", 90),
    Question("s03", "Two senior engineers disagree on the right architecture. You need to ship next week. What do you do?", "situational", "hard", 120),
    Question("s04", "You discover a serious bug an hour before launch. Walk me through your decision.", "situational", "hard", 90),
    Question("s05", "A teammate takes credit for your work in front of leadership. How do you handle it?", "situational", "hard", 90),

    # Strengths and self-awareness
    Question("p01", "What is your biggest strength, with a concrete example?", "personal", "easy", 90),
    Question("p02", "What is one thing you are actively working to improve?", "personal", "easy", 90),
    Question("p03", "What kind of work environment brings out your best?", "personal", "easy", 60),
    Question("p04", "How do you decide whether a job is worth taking?", "personal", "medium", 90),
    Question("p05", "Where do you want to be professionally in five years?", "personal", "medium", 90),

    # Closing
    Question("c01", "Why do you want to work here specifically?", "closing", "easy", 90),
    Question("c02", "Why are you leaving your current role?", "closing", "easy", 90),
    Question("c03", "What questions do you have for us?", "closing", "easy", 90),
]


CATEGORIES = ["behavioural", "leadership", "technical", "situational", "personal", "closing"]


def all_questions() -> list[dict]:
    return [_to_dict(q) for q in _QUESTIONS]


def by_category(category: str) -> list[dict]:
    cat = category.strip().lower()
    return [_to_dict(q) for q in _QUESTIONS if q.category == cat]


def random_set(n: int = 5, category: str | None = None, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    pool = [q for q in _QUESTIONS if (category is None or q.category == category)]
    rng.shuffle(pool)
    return [_to_dict(q) for q in pool[:max(1, n)]]


def get_question(qid: str) -> dict | None:
    for q in _QUESTIONS:
        if q.id == qid:
            return _to_dict(q)
    return None


def _to_dict(q: Question) -> dict:
    return {
        "id": q.id,
        "text": q.text,
        "category": q.category,
        "difficulty": q.difficulty,
        "target_seconds": q.target_seconds,
    }
