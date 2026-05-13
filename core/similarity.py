"""
core/similarity.py
──────────────────
Similarity utilities used by the Scoring Agent.

Provides:
  - cosine_similarity          : vector-space similarity
  - jaccard_similarity         : set-overlap for skill lists
  - skill_overlap_score        : % of JD skills present in candidate
  - hybrid_similarity          : weighted blend of semantic + lexical
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Vector similarity
# ──────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.
    Returns a value in [0, 1] (assuming normalized embeddings).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ──────────────────────────────────────────────
# Set / token similarity
# ──────────────────────────────────────────────

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Jaccard similarity between two sets.
    Returns |A ∩ B| / |A ∪ B|.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def skill_overlap_score(
    jd_skills: List[str],
    candidate_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Compute how many of the JD-required skills the candidate possesses.

    Returns:
        overlap_pct   : float in [0, 1]
        matched       : skills present in both lists
        missing       : JD skills not found in candidate
    """
    if not jd_skills:
        return 1.0, [], []

    # Normalise: lowercase + strip
    jd_set = {s.lower().strip() for s in jd_skills}
    cand_set = {s.lower().strip() for s in candidate_skills}

    # Soft match: allow substring matching for compound skills
    matched = []
    missing = []
    for jd_skill in jd_set:
        found = any(
            jd_skill in cand_skill or cand_skill in jd_skill
            for cand_skill in cand_set
        )
        if found:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    overlap_pct = len(matched) / len(jd_set)
    return overlap_pct, matched, missing


# ──────────────────────────────────────────────
# Hybrid similarity
# ──────────────────────────────────────────────

def hybrid_similarity(
    semantic_score: float,
    lexical_score: float,
    semantic_weight: float = 0.6,
    lexical_weight: float = 0.4,
) -> float:
    """
    Weighted blend of semantic (embedding-based) and lexical (skill-overlap) scores.

    Args:
        semantic_score : cosine similarity from embeddings, in [0, 1]
        lexical_score  : jaccard / overlap ratio, in [0, 1]
        semantic_weight: weight for semantic component (default 0.6)
        lexical_weight : weight for lexical component (default 0.4)

    Returns:
        Hybrid score in [0, 1]
    """
    assert abs(semantic_weight + lexical_weight - 1.0) < 1e-6, (
        "Weights must sum to 1.0"
    )
    return semantic_weight * semantic_score + lexical_weight * lexical_score


# ──────────────────────────────────────────────
# Score → rubric band conversion
# ──────────────────────────────────────────────

def similarity_to_rubric(
    similarity: float,
    low_threshold: float = 0.30,
    high_threshold: float = 0.85,
    low_band: Tuple[int, int] = (0, 3),
    mid_band: Tuple[int, int] = (4, 7),
    high_band: Tuple[int, int] = (8, 10),
) -> int:
    """
    Map a [0, 1] similarity score to a rubric integer score [0, 10].

    Default thresholds follow the problem spec:
      - < 30% similarity → low band (0–3)
      - 30–85% similarity → mid band (4–7)
      - > 85% similarity → high band (8–10)
    """
    if similarity < low_threshold:
        # Linearly interpolate within low band
        ratio = similarity / low_threshold
        return int(low_band[0] + ratio * (low_band[1] - low_band[0]))

    elif similarity < high_threshold:
        # Linearly interpolate within mid band
        ratio = (similarity - low_threshold) / (high_threshold - low_threshold)
        return int(mid_band[0] + ratio * (mid_band[1] - mid_band[0]))

    else:
        # Linearly interpolate within high band
        ratio = min((similarity - high_threshold) / (1.0 - high_threshold), 1.0)
        return int(high_band[0] + ratio * (high_band[1] - high_band[0]))
