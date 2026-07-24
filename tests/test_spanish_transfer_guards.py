from pathlib import Path

from autoresearch.score_mcq_ll import normalize_mcq_record
from prepare_dataset import _is_spanish_transfer_path


def test_spanish_transfer_paths_are_training_forbidden():
    assert _is_spanish_transfer_path(
        Path("../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/english_v5clean_benchmark_es.jsonl")
    )
    assert not _is_spanish_transfer_path(Path("data/sources/clean_mcq.jsonl"))


def test_transfer_records_keep_stable_source_id_and_metadata():
    row = {
        "id": "es-transfer-mcq-ag-0001",
        "source_question_id": "AG-0001",
        "language": "es",
        "locale": "es-419",
        "benchmark_role": "en_to_es_parallel_language_transfer_diagnostic",
        "question": "¿Qué servicio ecosistémico se menciona?",
        "A": "Control de la erosión",
        "B": "Dependencia de fertilizantes",
        "C": "Monocultivo",
        "D": "Eliminación de organismos",
        "answer": "A",
        "source_metadata": {
            "category": "Agroecology Benchmark",
            "question_topic": "ecosystem_services",
            "source_benchmark_sha256": "a" * 64,
        },
    }
    item = normalize_mcq_record(row, 1)
    assert item["question_id"] == "AG-0001"
    assert item["source_question_id"] == "AG-0001"
    assert item["language"] == "es"
    assert item["benchmark_role"] == "en_to_es_parallel_language_transfer_diagnostic"
    assert item["category"] == "Agroecology Benchmark"
