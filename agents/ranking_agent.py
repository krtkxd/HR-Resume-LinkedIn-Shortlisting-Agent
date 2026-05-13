"""
agents/ranking_agent.py
────────────────────────
Ranking Agent

Sorts a list of scored candidates by total_score (descending).
Adds rank, percentile, and tier classification to each result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TIER_THRESHOLDS = {
    "A": 8.0,   # Top tier — Strong Hire
    "B": 6.5,   # Strong candidates — Hire
    "C": 5.0,   # Borderline — Maybe
    "D": 0.0,   # Below threshold — No Hire
}


class RankingAgent:
    """
    Ranks a list of scored candidate result dicts by total_score.

    Usage:
        agent = RankingAgent()
        ranked = agent.rank(scored_candidates, top_n=10)
        stats  = agent.summary_stats(ranked)
    """

    def rank(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sort candidates by total_score descending and enrich with rank metadata.

        Args:
            scored_candidates : list of dicts from ScoringAgent.score()
            top_n             : if set, return only the top N candidates

        Returns:
            Ranked list with added fields: rank, percentile, tier
        """
        if not scored_candidates:
            return []

        # Sort descending by total_score; tie-break on name for determinism
        sorted_cands = sorted(
            scored_candidates,
            key=lambda c: (-c["total_score"], c.get("name", "")),
        )
        n = len(sorted_cands)

        for idx, cand in enumerate(sorted_cands, start=1):
            cand["rank"] = idx
            # Percentile: rank 1 → 100th, rank n → (1/n)*100
            cand["percentile"] = round((1 - (idx - 1) / n) * 100, 1)
            cand["tier"] = self._assign_tier(cand["total_score"])

        return sorted_cands[:top_n] if top_n else sorted_cands

    def _assign_tier(self, score: float) -> str:
        """Map a total score to a letter tier (A/B/C/D)."""
        for tier, threshold in sorted(TIER_THRESHOLDS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return tier
        return "D"

    def summary_stats(self, ranked: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Return aggregate statistics for the ranked candidate list.

        Returns:
            Dict with total_candidates, average_score, highest_score,
            lowest_score, tier_distribution, hire_count.
        """
        if not ranked:
            return {
                "total_candidates": 0,
                "average_score": 0.0,
                "highest_score": 0.0,
                "lowest_score": 0.0,
                "tier_distribution": {},
                "hire_count": 0,
            }

        scores = [c["total_score"] for c in ranked]

        tier_counts: Dict[str, int] = {}
        for c in ranked:
            t = c.get("tier", "D")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        hire_count = sum(
            1 for c in ranked
            if c.get("recommendation", "") in ("Hire", "Strong Hire")
        )

        return {
            "total_candidates": len(ranked),
            "average_score": round(sum(scores) / len(scores), 2),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "tier_distribution": tier_counts,
            "hire_count": hire_count,
        }
