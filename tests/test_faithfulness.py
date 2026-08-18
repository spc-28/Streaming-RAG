"""The rule-based layers of the faithfulness check (the NLI model is not loaded here)."""
from app.faithfulness import (
    _citation_accuracy, _clean_claim, _extract_claims, _extract_numbers, _numeric_accuracy, split_sentences,
)


def test_sentence_splitter_keeps_abbreviations():
    assert len(split_sentences("First sentence. Second one! And third?")) == 3
    assert len(split_sentences("Dr. Smith met Mr. Jones. They discussed the U.S. economy.")) == 2


def test_extract_numbers():
    nums = _extract_numbers("The 50 million people. About 3.5 percent. Roughly 1,000 events.")
    assert nums == {"50m", "3.5%", "1000"}
    # the scale matters: "50 billion" is not supported by a source that says "50 million"
    assert _numeric_accuracy("It cost 50 billion.", "It cost 50 million.") == 0.0


def test_numeric_accuracy():
    assert _numeric_accuracy("100 attendees and 5 speakers.", "It had 100 attendees and 5 speakers.") == 1.0
    assert 0 < _numeric_accuracy("100 attendees and 999 speakers.", "It had 100 attendees.") < 1
    assert _numeric_accuracy("Just words.", "Other words.") == 1.0  # nothing to verify
    # citation times and story numbers are references, not numeric claims
    assert _numeric_accuracy("In the 07:08 UTC segment (story 2), 5 people spoke.", "5 people spoke.") == 1.0


def test_citation_accuracy():
    starts = ["2026-01-15T09:12:00+00:00"]
    assert _citation_accuracy("In the 09:12 UTC segment, the vote passed.", starts) == 1.0
    assert _citation_accuracy("In the 23:45 UTC segment, big news.", starts) == 0.0
    assert _citation_accuracy("No citation here.", starts) == 1.0


def test_claims_drop_citation_preamble_and_meta_sentences():
    assert _clean_claim("In the 09:12 UTC segment, it is mentioned that X happened.") == "X happened."
    assert _clean_claim("In the 07:08 UTC segment (story 2), NASA found water.") == "NASA found water."
    assert _clean_claim("Initially, in the 07:08 UTC segment (story 1), rates held.") == "Initially, rates held."
    claims = _extract_claims("X happened yesterday. Based on the transcripts, this is unrelated. "
                             "Y also occurred. I don't have enough information.")
    assert claims == ["X happened yesterday.", "Y also occurred."]
