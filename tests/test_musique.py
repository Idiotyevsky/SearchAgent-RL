import json

import pytest

from efficienttool_rl.data import load_musique, load_verl_examples
from efficienttool_rl.evaluation import answer_metrics_any
from efficienttool_rl.training import to_verl_record
from scripts.evaluate import _metrics_summary


def row(*, answerable=True):
    return {
        "id": "3hop__q1_q2_q3",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Bridge A",
                "paragraph_text": "Bridge A identifies Bridge B.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Distractor",
                "paragraph_text": "This paragraph is unrelated.",
                "is_supporting": False,
            },
            {
                "idx": 2,
                "title": "Answer",
                "paragraph_text": "The final answer is Paris.",
                "is_supporting": True,
            },
            {
                "idx": 3,
                "title": "Bridge B",
                "paragraph_text": "Bridge B points to the answer document.",
                "is_supporting": True,
            },
        ],
        "question": "Which city completes the chain?",
        "question_decomposition": [
            {"id": 0, "question": "first", "answer": "b", "paragraph_support_idx": 0},
            {"id": 1, "question": "second", "answer": "c", "paragraph_support_idx": 3},
            {"id": 2, "question": "third", "answer": "Paris", "paragraph_support_idx": 2},
        ],
        "answer": "Paris",
        "answer_aliases": ["City of Paris", "Paris"],
        "answerable": answerable,
    }


def test_load_official_musique_shape_and_hop_metadata(tmp_path):
    path = tmp_path / "musique_dev.jsonl"
    path.write_text(json.dumps(row()) + "\n", encoding="utf-8")

    example = load_musique(path, split="dev")[0]

    assert example.example_id == "3hop__q1_q2_q3"
    assert example.hop_count == 3
    assert example.level == "3hop"
    assert example.dataset_name == "musique"
    assert example.supporting_titles == ("Bridge A", "Bridge B", "Answer")
    assert example.answer_aliases == ("City of Paris", "Paris")
    assert len(example.passages) == 4


def test_musique_record_keeps_gold_metadata_out_of_model_context(tmp_path):
    path = tmp_path / "musique_dev.jsonl"
    path.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    example = load_musique(path, split="dev")[0]

    record = to_verl_record(
        example,
        index=0,
        max_top_k=1,
        max_executed_search_calls=4,
        data_source="musique_ans_local",
    )

    kwargs = record["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]
    assert record["extra_info"]["hop_count"] == 3
    assert record["extra_info"]["answer_aliases"] == ["City of Paris", "Paris"]
    assert record["extra_info"]["supporting_titles"] == ["Bridge A", "Bridge B", "Answer"]
    assert kwargs["max_executed_search_calls"] == 4
    assert "supporting_titles" not in json.dumps(record["prompt"])
    assert "answer_aliases" not in json.dumps(record["prompt"])
    assert "supporting_titles" not in json.dumps(kwargs)
    assert "answer_aliases" not in json.dumps(kwargs)


def test_musique_loader_skips_unanswerable_rows_by_default(tmp_path):
    path = tmp_path / "musique_full_dev.jsonl"
    path.write_text(json.dumps(row(answerable=False)) + "\n", encoding="utf-8")
    assert load_musique(path, split="dev") == []


def test_musique_loader_rejects_support_alignment_mismatch(tmp_path):
    broken = row()
    broken["paragraphs"][3]["is_supporting"] = False
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="support disagree"):
        load_musique(path, split="dev")


def test_answer_alias_is_accepted():
    assert answer_metrics_any("City of Paris", ("Paris", "City of Paris")) == {
        "exact_match": 1.0,
        "f1": 1.0,
    }


def test_musique_metadata_round_trips_through_verl_parquet(tmp_path):
    datasets = pytest.importorskip("datasets")
    source = tmp_path / "musique_dev.jsonl"
    source.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    record = to_verl_record(
        load_musique(source, split="dev")[0],
        index=0,
        max_executed_search_calls=4,
        data_source="musique_ans_local",
    )
    parquet = tmp_path / "musique.parquet"
    datasets.Dataset.from_list([record]).to_parquet(str(parquet))

    loaded = load_verl_examples(parquet, split="dev")[0]

    assert loaded.dataset_name == "musique"
    assert loaded.hop_count == 3
    assert loaded.answer_aliases == ("City of Paris", "Paris")
    assert loaded.supporting_titles == ("Bridge A", "Bridge B", "Answer")


def test_evaluation_summary_is_stratified_by_required_hops():
    def record(hops, em, searches):
        return {
            "hop_count": hops,
            "exact_match": em,
            "f1": em,
            "task_reward": em,
            "executed_search_calls": searches,
            "trajectory": {
                "termination_reason": "final_answer",
                "attempted_tool_calls": searches,
                "valid_tool_calls": searches,
                "executed_tool_calls": searches,
                "executed_search_calls": searches,
                "invalid_actions": 0,
                "steps": [{}] * (searches + 1),
            },
        }

    summary = _metrics_summary([record(2, 1.0, 2), record(4, 0.0, 1)])

    assert summary["episodes"] == 2
    assert summary["by_hop"]["2"]["em"] == 1.0
    assert summary["by_hop"]["2"]["multi_search_rate"] == 1.0
    assert summary["by_hop"]["4"]["em"] == 0.0
    assert summary["by_hop"]["4"]["multi_search_rate"] == 0.0


def test_duplicate_musique_titles_receive_stable_document_labels(tmp_path):
    duplicate = row()
    duplicate["paragraphs"][3]["title"] = "Bridge A"
    path = tmp_path / "duplicate_titles.jsonl"
    path.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")

    example = load_musique(path, split="dev")[0]

    assert "Bridge A [MuSiQue paragraph 0]" in example.supporting_titles
    assert "Bridge A [MuSiQue paragraph 3]" in example.supporting_titles
    assert len(example.supporting_titles) == example.hop_count
