"""
Multi-granularity faithfulness: is the answer actually grounded in the sources?

  1. Claims     each answer sentence is checked by an NLI model against every source sentence
  2. Numbers    every number in the answer must appear somewhere in the sources
  3. Citations  every "HH:MM UTC" the answer cites must belong to a retrieved chunk

Claim-level NLI (what RAGAS / DeepEval do) misses 2 and 3, yet a wrong figure
or a made-up citation is the most damaging error in a news answer.
"""
import re
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.models import Chunk, ClaimVerdict, FaithfulnessReport

_ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "U.S.", "U.K.", "U.N.", "E.U.", "St.", "vs.", "No."]


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that does not break on common abbreviations."""
    for a in _ABBREVIATIONS:
        text = text.replace(a, a.replace(".", "<DOT>"))
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if len(p.strip()) > 5]


@lru_cache
def _nli():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(get_settings().nli_model)
    labels = {name.lower(): int(i) for i, name in model.model.config.id2label.items()}
    return model, labels["entailment"], labels["contradiction"]


# ---------- 1. Claims ----------
_META_SENTENCE = re.compile(
    r"(?i)based on.*transcripts|i don'?t have enough|transcripts? (do not|don'?t)|no .*information")


_REFERENCES = [  # citation scaffolding the prompt asks for; it is not part of the factual claim
    r"\s*\([^()]*\b(UTC|story|segment)\b[^()]*\)",                                  # "(story 2)", "(09:12 UTC)"
    r"\b(in|according to|from|during|at)\s+(the\s+)?\d{1,2}[.:]\d{2}(:\d{2})?\s*(UTC)?"
    r"\s*(segment|section|part|transcript)?\s*,?\s*",                               # "In the 09:12 UTC segment, "
    r"\b\d{1,2}[.:]\d{2}\s*UTC\b",                                                  # any other "09:12 UTC"
    r"\bstory\s+\d+\b",
]


def _strip_references(text: str) -> str:
    for pattern in _REFERENCES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_claim(claim: str) -> str:
    claim = _strip_references(claim)
    claim = re.sub(r"(?i)^it\s+is\s+(mentioned|stated|noted|said)\s+that\s+", "", claim)
    return claim[:1].upper() + claim[1:]


def _extract_claims(answer: str) -> list[str]:
    claims = (_clean_claim(s) for s in split_sentences(answer))
    return [c for c in claims if len(c) >= 10 and not _META_SENTENCE.search(c)]


def _verify_claims(claims: list[str], source_sentences: list[str]) -> list[ClaimVerdict]:
    model, ENT, CON = _nli()
    pairs = [(src, claim) for claim in claims for src in source_sentences]
    logits = np.asarray(model.predict(pairs)).reshape(len(claims), len(source_sentences), -1)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)

    verdicts = []
    for claim, p in zip(claims, probs):
        # Judge each claim against its best-matching source sentence. Taking the max
        # contradiction over *all* sentences would flag claims that unrelated sentences "contradict".
        best = p[:, ENT].argmax()
        entail, contra = float(p[best, ENT]), float(p[best, CON])
        if contra > 0.7 and contra > entail:
            verdict = "contradicted"
        elif entail > 0.5:
            verdict = "supported"
        elif entail > 0.3:
            verdict = "partially_supported"
        else:
            verdict = "unsupported"
        verdicts.append(ClaimVerdict(claim=claim, verdict=verdict, entailment_score=round(entail, 3)))
    return verdicts


# ---------- 2. Numbers ----------
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s?(%|percent|million|billion|thousand|k|m|b))?(?!\w)",
    re.IGNORECASE)
_UNITS = {"percent": "%", "million": "m", "billion": "b", "thousand": "k"}


def _extract_numbers(text: str) -> set[str]:
    """Numbers with their scale, normalised: "1,000" → "1000", "50 million" → "50m", "4.5 percent" → "4.5%"."""
    return {num.replace(",", "") + _UNITS.get(unit.lower(), unit.lower())
            for num, unit in _NUMBER.findall(text)}


def _numeric_accuracy(answer: str, source_text: str) -> float:
    """Fraction of the answer's numbers that also appear in the sources (1.0 if it has none)."""
    nums = _extract_numbers(_strip_references(answer))  # "09:12 UTC" and "story 2" are not claims
    if not nums:
        return 1.0
    return round(len(nums & _extract_numbers(source_text)) / len(nums), 3)


# ---------- 3. Citations ----------
_CITATION = re.compile(r"\b(\d{1,2})[.:](\d{2})(?:[.:]\d{2})?\s*UTC", re.IGNORECASE)


def _citation_accuracy(answer: str, start_times: list[str]) -> float:
    """Fraction of cited HH:MM UTC times that match a retrieved chunk's ISO start time."""
    cited = _CITATION.findall(answer)
    if not cited:
        return 1.0
    available = {t[11:16] for t in start_times}
    return round(sum(f"{h.zfill(2)}:{m}" in available for h, m in cited) / len(cited), 3)


def check_faithfulness(answer: str, docs: list[Chunk]) -> FaithfulnessReport:
    claims = _extract_claims(answer)
    source_sentences = [s for d in docs for s in split_sentences(d.text)]
    if not claims:  # e.g. "I don't have enough information" — nothing to verify
        return FaithfulnessReport(faithfulness_score=1.0)
    if not source_sentences:
        return FaithfulnessReport(faithfulness_score=0.0, total_claims=len(claims), unsupported=len(claims))

    results = _verify_claims(claims, source_sentences)
    count = {v: sum(r.verdict == v for r in results)
             for v in ("supported", "partially_supported", "unsupported", "contradicted")}
    return FaithfulnessReport(
        faithfulness_score=round((count["supported"] + 0.5 * count["partially_supported"]) / len(results), 3),
        total_claims=len(results),
        **count,
        numeric_accuracy=_numeric_accuracy(answer, " ".join(d.text for d in docs)),
        citation_accuracy=_citation_accuracy(answer, [d.metadata.get("start_time_utc", "") for d in docs]),
        claim_results=results,
    )
