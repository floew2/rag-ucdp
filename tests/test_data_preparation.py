"""Tests for the UCDP download and the raw-CSV → parquet preparation."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import rag_ucdp.data_loading as dl
from rag_ucdp.exceptions import DataValidationError

DERIVED = ["type_of_violence_label", "source_office_list"]


def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_download_url_pattern() -> None:
    assert dl.ged_download_url("25.1") == "https://ucdp.uu.se/downloads/ged/ged251-csv.zip"
    assert dl.ged_download_url("26.1").endswith("/ged261-csv.zip")


def test_download_extracts_csv_and_reuses_zip(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes({"GEDEvent_v25_1.csv": "id,year\n1,2022\n"})
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float = 0) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(dl, "urlopen", fake_urlopen)
    csv_path = dl.download_ged_csv("25.1", tmp_path)

    assert csv_path == tmp_path / "GEDEvent_v25_1.csv"
    assert csv_path.read_text() == "id,year\n1,2022\n"
    assert calls == ["https://ucdp.uu.se/downloads/ged/ged251-csv.zip"]

    # Second run uses the zip on disk instead of downloading again.
    dl.download_ged_csv("25.1", tmp_path)
    assert len(calls) == 1


def test_download_rejects_zip_without_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        dl, "urlopen", lambda url, timeout=0: io.BytesIO(_zip_bytes({"readme.txt": "x"}))
    )
    with pytest.raises(DataValidationError):
        dl.download_ged_csv("25.1", tmp_path)


def test_prepare_processed_reproduces_published_table(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Raw-format rows go through filter + derive and come out like the parquet."""
    raw = sample_df.drop(columns=DERIVED).copy()
    outside = raw.iloc[[0]].copy()
    outside["side_a"] = "Government of Sudan"
    outside["side_b"] = "RSF"
    raw = pd.concat([raw, outside], ignore_index=True)
    csv_path = tmp_path / "ged.csv"
    raw.to_csv(csv_path, index=False)

    out = tmp_path / "processed" / "events.parquet"
    df = dl.prepare_processed(
        csv_path,
        out,
        conflict_parties_include=["Ukraine", "Government of Ukraine"],
        date_start="2022-01-01",
        date_end="2024-12-31",
    )

    assert out.exists()
    assert len(df) == len(sample_df)  # the out-of-scope row is dropped
    assert list(df.columns) == list(sample_df.columns)
    assert sorted(df["id"]) == sorted(sample_df["id"])
