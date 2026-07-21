"""The match rule must tolerate cosmetic differences but reject non-matches."""
from src.eval.match import chunk_contains_passage, normalize


def test_normalize_strips_punct_case_and_entities():
    assert normalize("Atrial Fibrillation, 4 g/day!") == "atrial fibrillation 4 g day"
    assert normalize("up&nbsp;to 1&nbsp;in&nbsp;10") == "up to 1 in 10"


def test_match_tolerates_tokenizer_spacing_and_case():
    # Chunk text as the uncased embedding tokenizer would render it.
    chunk = "the observed risk was found to be highest with a dose of 4 g / day"
    passage = "the observed risk was found to be highest with a dose of 4 g/day"
    assert chunk_contains_passage(chunk, passage)


def test_match_token_overlap_when_split_across_boundary():
    # Passage mostly present (>=0.9 of tokens) but not a clean substring.
    chunk = "atrial fibrillation is now listed as an adverse drug reaction common frequency"
    passage = "atrial fibrillation is now listed as an adverse drug reaction"
    assert chunk_contains_passage(chunk, passage)


def test_match_rejects_unrelated_text():
    chunk = "warfarin can interact with tramadol increasing the risk of bleeding"
    passage = "topiramate is contraindicated in pregnancy for migraine prophylaxis"
    assert not chunk_contains_passage(chunk, passage)
