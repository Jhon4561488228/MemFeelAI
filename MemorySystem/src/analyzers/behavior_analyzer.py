"""
Behavior analyzer for Procedural Memory.
Analyzes practice patterns and produces simple recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime


def analyze_patterns(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate basic behavior patterns from procedural skill items.

    Each item is expected to contain metadata keys: skill_type, practice_count,
    success_rate, last_practiced (ISO), difficulty, proficiency.
    """
    patterns: Dict[str, Any] = {
        "total_skills": 0,
        "practice_by_hour": {str(h): 0 for h in range(24)},
        "by_skill_type": {},
        "avg_success": 0.0,
        "avg_proficiency": 0.0,
        "hard_skills_low_prof": 0,
    }

    if not items:
        return patterns

    total_success = 0.0
    total_prof = 0.0
    patterns["total_skills"] = len(items)

    for meta in items:
        # hour histogram
        ts = meta.get("last_practiced")
        try:
            hour = datetime.fromisoformat(ts).hour if ts else None
        except Exception:
            hour = None
        if hour is not None:
            patterns["practice_by_hour"][str(hour)] += 1

        # skill type counts
        st = meta.get("skill_type", "general")
        patterns["by_skill_type"][st] = patterns["by_skill_type"].get(st, 0) + 1

        # averages
        total_success += float(meta.get("success_rate", 0.0))
        total_prof += float(meta.get("proficiency", 0.0))

        # difficult skills with low proficiency
        if float(meta.get("difficulty", 0.5)) >= 0.7 and float(meta.get("proficiency", 0.0)) < 0.4:
            patterns["hard_skills_low_prof"] += 1

    patterns["avg_success"] = total_success / max(1, len(items))
    patterns["avg_proficiency"] = total_prof / max(1, len(items))
    return patterns


def build_recommendations(patterns: Dict[str, Any]) -> List[str]:
    """Generate simple recommendations from patterns.
    Heuristics are intentionally lightweight and deterministic.
    """
    recs: List[str] = []

    # Best hour to practice
    if patterns.get("practice_by_hour"):
        best_hour = max(patterns["practice_by_hour"].items(), key=lambda kv: kv[1])[0]
        if patterns["practice_by_hour"][best_hour] > 0:
            recs.append(f"Практикуйтесь чаще около {best_hour}:00 — у вас там наибольшая активность.")

    # Low average success
    if patterns.get("avg_success", 0.0) < 0.5 and patterns.get("total_skills", 0) > 0:
        recs.append("Средний успех < 50%. Попробуйте уменьшить сложность или увеличить интервалы отдыха.")

    # Hard skills with low proficiency
    if patterns.get("hard_skills_low_prof", 0) > 0:
        recs.append("Есть сложные навыки с низким прогрессом — выделите фокус‑сессии и разбейте на поднавыки.")

    # Focus suggestion by dominant type
    if patterns.get("by_skill_type"):
        dominant = max(patterns["by_skill_type"].items(), key=lambda kv: kv[1])[0]
        recs.append(f"Сфокусируйтесь на типе навыков '{dominant}' для ускорения прогресса.")

    return recs

