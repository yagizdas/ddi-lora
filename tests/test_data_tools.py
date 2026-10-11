import json
import random

import pytest

from patient_aware_ddi.merge import combine_jsonl_by_url
from patient_aware_ddi.negatives import generate_negatives, pair_key
from patient_aware_ddi.sample import sample_records


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_merge_first_source_wins_and_counts_duplicates(tmp_path):
    first, second, output = [tmp_path / name for name in ["a.jsonl", "b.jsonl", "merged.jsonl"]]
    write_rows(first, [{"url": "one", "value": "first"}, {"url": "two"}])
    write_rows(second, [{"url": "one", "value": "later"}, {"url": "three"}])
    stats = combine_jsonl_by_url([str(first), str(second)], str(output))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert stats == {"written": 3, "skipped_duplicates": 1, "unique_urls": 3}
    assert rows[0]["value"] == "first"


@pytest.mark.parametrize("invalid", ['{"url":', "{}", "[]", '{"url": 5}'])
def test_merge_preserves_existing_output_on_invalid_input(tmp_path, invalid):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text('{"url":"valid"}\n' + invalid + "\n")
    output.write_text("existing result")
    with pytest.raises(ValueError):
        combine_jsonl_by_url([str(source)], str(output))
    assert output.read_text() == "existing result"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.jsonl", "output.jsonl"]


def test_merge_rejects_input_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    write_rows(source, [{"url": "one"}])
    with pytest.raises(ValueError, match="differ"):
        combine_jsonl_by_url([str(source)], str(source))
    assert json.loads(source.read_text())["url"] == "one"


def test_source_negatives_exclude_known_and_self_pairs():
    rows = [{"drug_a": "Alpha", "drug_b": "Beta"}, {"drug_a": " beta ", "drug_b": "Gamma"}]
    negatives = generate_negatives(rows, 1, seed=42)
    assert len(negatives) == 1
    assert pair_key(negatives[0]["drug_a"], negatives[0]["drug_b"]) == ("alpha", "gamma")
    assert negatives[0]["label"] == "NO_SOURCE_SUPPORTED_DDI"
    assert negatives[0]["source_status"] == "no_record_in_project_source"


def test_source_negatives_impossible_count_is_explicit():
    with pytest.raises(ValueError, match="only 0"):
        generate_negatives([{"drug_a": "Alpha", "drug_b": "Beta"}], 1)
    assert generate_negatives([], 0) == []


def test_source_negative_sampling_is_reproducible_without_global_rng_changes():
    rows = [{"drug_a": "Alpha", "drug_b": drug} for drug in ["Beta", "Gamma", "Delta", "Epsilon"]]
    state = random.getstate()
    assert generate_negatives(rows, 3, 7) == generate_negatives(rows, 3, 7)
    assert random.getstate() == state


def test_stratified_sampling_counts_and_shortfall():
    rows = [
        {"id": i, "metadata": {"label": label}} for i, label in enumerate(["A"] * 5 + ["B"] * 3)
    ]
    sample = sample_records(rows, {"A": 2, "B": 1}, random.Random(42))
    assert len(sample) == 3
    assert sum(r["metadata"]["label"] == "A" for r in sample) == 2
    assert len({r["id"] for r in sample}) == 3
    assert sample == sample_records(rows, {"A": 2, "B": 1}, random.Random(42))
    with pytest.raises(ValueError, match="only found 3"):
        sample_records(rows, {"B": 4}, random.Random(42))
