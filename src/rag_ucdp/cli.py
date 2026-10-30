"""Command-line interface: ``rag-ucdp <command>``.

Commands, in the order a new user runs them:

- ``check`` — show what is in place and what is missing.
- ``data``  — download UCDP GED and rebuild the processed event table
  (optional: the table for the configured release ships with the repository).
- ``index`` — build the vector and BM25 indexes (needed once).
- ``prepare`` — build missing indexes and download a missing answer model;
  skips whatever is already there (used by the Docker setup).
- ``app``   — start the Streamlit web app.
- ``ask``   — answer one question in the terminal (text or JSON).

All settings come from ``config/config.yaml``; ``--config`` points elsewhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from rag_ucdp.config import DEFAULT_CONFIG_PATH, REPO_ROOT, AppConfig, load_config, resolve_path

logger = logging.getLogger("rag_ucdp")

#: Streamlit entry script, relative to the repository root.
APP_SCRIPT = REPO_ROOT / "app" / "streamlit_app.py"

#: UCDP release the published evaluation results refer to.
EVALUATED_GED_VERSION = "25.1"


# --- Commands ---


def cmd_check(cfg: AppConfig, _args: argparse.Namespace) -> int:
    """Print readiness of data, indexes and the answer model."""
    from rag_ucdp.pipeline import check_status

    items = check_status(cfg)
    for item in items:
        mark = "ok " if item.ok else ("!! " if item.required else "-- ")
        line = f"[{mark}] {item.name:<13} {item.detail}"
        if not item.ok and item.fix:
            line += f"  →  {item.fix}"
        print(line)
    missing = [i for i in items if i.required and not i.ok]
    if missing:
        print("\nNot ready: run the commands shown after → above.")
        return 1
    if not all(i.ok for i in items):
        print("\nReady for structured questions; semantic answers need the LLM (see above).")
    else:
        print("\nAll set. Start the app with: rag-ucdp app")
    return 0


def cmd_data(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Download UCDP GED (or use a local CSV) and write the processed table."""
    from rag_ucdp.data_loading import download_ged_csv, prepare_processed

    version = args.version or cfg.data.ged_version
    output = resolve_path(cfg.data.processed_file)
    if output.exists() and not args.force:
        print(
            f"{cfg.data.processed_file} already exists. Use --force to rebuild it "
            "(then rebuild the indexes with `rag-ucdp index --force`)."
        )
        return 0

    if args.csv:
        csv_path = Path(args.csv)
    else:
        print(f"Downloading UCDP GED {version} …")
        csv_path = download_ged_csv(version, resolve_path(cfg.data.raw_dir))

    df = prepare_processed(
        csv_path,
        output,
        conflict_parties_include=cfg.scope.conflict_parties_include,
        date_start=cfg.scope.date_start.isoformat(),
        date_end=cfg.scope.date_end.isoformat(),
    )
    print(f"Wrote {len(df):,} events to {cfg.data.processed_file}.")
    if version != EVALUATED_GED_VERSION:
        print(
            f"Note: the evaluation results in the README refer to GED "
            f"{EVALUATED_GED_VERSION}; they do not carry over to {version}."
        )
    print("Next: rag-ucdp index --force")
    return 0


def cmd_index(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Build the ChromaDB and BM25 indexes from the processed table."""
    from rag_ucdp.indexing import indexes_exist

    chroma_dir = resolve_path(cfg.data.chroma_dir)
    bm25_path = resolve_path(cfg.data.bm25_path)
    if indexes_exist(chroma_dir, cfg.data.collection_name, bm25_path) and not args.force:
        print("Indexes already exist. Use --force to rebuild them.")
        return 0
    code = _build_indexes(cfg, args.device, args.batch_size)
    if code == 0:
        print("Next: rag-ucdp app")
    return code


def _build_indexes(cfg: AppConfig, device: str | None, batch_size: int) -> int:
    from rag_ucdp.data_loading import load_processed
    from rag_ucdp.embeddings import EmbeddingModel
    from rag_ucdp.indexing import build_indexes

    chroma_dir = resolve_path(cfg.data.chroma_dir)
    bm25_path = resolve_path(cfg.data.bm25_path)

    parquet = resolve_path(cfg.data.processed_file)
    if not parquet.exists():
        print(f"{cfg.data.processed_file} not found. Run `rag-ucdp data` first.")
        return 1

    df = load_processed(parquet)
    device = device or cfg.models.embedding_device
    print(
        f"Embedding {len(df):,} events with {cfg.models.embedding_model} on {device}. "
        "The first run downloads the model (about 1.3 GB); on CPU this takes hours."
    )
    stats = build_indexes(
        df,
        EmbeddingModel(model_name=cfg.models.embedding_model, device=device),
        chroma_dir,
        cfg.data.collection_name,
        bm25_path,
        batch_size=batch_size,
        progress=lambda line: print(line, flush=True),
    )
    print(
        f"Indexed {stats.chroma_count:,} events (vector) and {stats.bm25_count:,} (BM25) "
        f"in {stats.total_seconds / 60:.1f} min."
    )
    return 0


def cmd_prepare(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Get everything ready: build missing indexes and pull a missing answer model.

    Safe to run on every start: steps that are already done are skipped. The
    Docker setup runs it before the app starts. Only a failed index build is
    an error; without the LLM the app still answers structured questions and
    shows what is missing, so model problems are reported but do not fail.
    """
    from rag_ucdp.config import ollama_base_url
    from rag_ucdp.indexing import indexes_exist
    from rag_ucdp.pipeline import ollama_models, pull_model, wait_for_ollama

    chroma_dir = resolve_path(cfg.data.chroma_dir)
    bm25_path = resolve_path(cfg.data.bm25_path)
    if indexes_exist(chroma_dir, cfg.data.collection_name, bm25_path):
        print("Indexes: ready.")
    else:
        print("Indexes: not built yet. Building them now (one time only).")
        code = _build_indexes(cfg, None, 32)
        if code != 0:
            return code

    if args.skip_model:
        return 0
    url, model = ollama_base_url(cfg), cfg.models.llm_model
    print(f"Ollama: waiting for {url} …")
    if not wait_for_ollama(url, timeout_s=args.wait):
        print(
            f"Warning: Ollama is not reachable at {url}. Structured questions will "
            "work; semantic answers need Ollama."
        )
        return 0
    if model in (ollama_models(url) or []):
        print(f"Answer model: {model} ready.")
        return 0
    print(f"Answer model: downloading {model} (one time only, several GB) …")
    if not pull_model(url, model):
        print(f"Warning: could not download {model}; semantic answers need it.")
        return 0
    print(f"Answer model: {model} ready.")
    return 0


def cmd_app(_cfg: AppConfig, args: argparse.Namespace) -> int:
    """Start the Streamlit app; extra arguments are passed to Streamlit."""
    command = [sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT), *args.streamlit_args]
    return subprocess.call(command)  # noqa: S603 - fixed executable and script


def cmd_ask(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Answer one question in the terminal."""
    from rag_ucdp.pipeline import PATH_LABELS, answer, check_status, load_resources

    missing = [i for i in check_status(cfg) if i.required and not i.ok]
    if missing:
        for item in missing:
            print(f"{item.name}: {item.detail}  →  {item.fix}")
        return 1
    result = answer(args.question, load_resources(cfg, warm_up=False))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 1 if result.error else 0

    print(f"Path:    {PATH_LABELS[result.path]}")
    if result.decision.filters:
        print(f"Filters: {result.decision.filters}")
    print()
    print(textwrap.fill(result.text, width=100) if "\n" not in result.text else result.text)
    for note in result.notes:
        print(f"\n{note}")
    if result.events:
        cited = set(result.cited_ids)
        print("\nEvidence:")
        for r in result.events[:10]:
            mark = "✓" if r.id in cited else " "
            print(
                f"  {mark} event {r.id} · {r.metadata.get('date_start', '?')} · "
                f"{r.metadata.get('adm_1', '?')}"
            )
    if result.source_bias is not None and result.source_bias.total_outlet_mentions:
        print(f"\nSources: {result.source_bias.description}")
    return 1 if result.error else 0


# --- Parser ---


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="rag-ucdp",
        description="Routed hybrid RAG over UCDP conflict-event data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("check", help="show what is ready and what is missing")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("data", help="download UCDP GED and rebuild the processed event table")
    p.add_argument("--version", help="GED release, e.g. 25.1 (default: from config)")
    p.add_argument("--csv", help="use this local GED CSV instead of downloading")
    p.add_argument("--force", action="store_true", help="overwrite the existing table")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("index", help="build the vector and BM25 indexes")
    p.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], help="embedding device (default: config)"
    )
    p.add_argument("--batch-size", type=int, default=32, help="embedding batch size")
    p.add_argument("--force", action="store_true", help="rebuild existing indexes")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("prepare", help="build missing indexes and download a missing answer model")
    p.add_argument(
        "--wait", type=float, default=120.0, help="seconds to wait for Ollama (default 120)"
    )
    p.add_argument("--skip-model", action="store_true", help="only build the indexes")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("app", help="start the web app (extra args go to Streamlit)")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("ask", help="answer one question in the terminal")
    p.add_argument("question", help="the question, in quotes")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.set_defaults(func=cmd_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rag-ucdp`` command."""
    parser = build_parser()
    # Unknown options are allowed only for `app`, which forwards them to Streamlit.
    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "app":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.streamlit_args = extra
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    code: int = args.func(cfg, args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
