"""
Evaluation from the command line.

  python -m app.cli eval      --dataset sample --pipeline full [--judge]
  python -m app.cli benchmark --dataset sample [--pipelines baseline hybrid full] [--judge]
  python -m app.cli synthetic --n-chunks 10 --questions 3
  python -m app.cli compare   BASELINE_RUN_ID CANDIDATE_RUN_ID   (exit code 1 on regression)
  python -m app.cli pareto    [--metric faithfulness] [--floor 0.7]
"""
import argparse
import sys
from typing import get_args

from app import evaluation as ev
from app.config import setup_logging
from app.models import PipelineName


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = [max([len(c)] + [len(_fmt(r.get(c))) for r in rows]) for c in columns]
    print("\n" + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(columns, widths)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming RAG evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    pipelines = list(get_args(PipelineName))

    p = sub.add_parser("eval", help="evaluate one pipeline")
    p.add_argument("--dataset", default="sample")
    p.add_argument("--pipeline", default="full", choices=pipelines)
    p.add_argument("--judge", action="store_true", help="also score answers with LLM-as-judge")

    p = sub.add_parser("benchmark", help="compare pipelines on one dataset")
    p.add_argument("--dataset", default="sample")
    p.add_argument("--pipelines", nargs="+", choices=pipelines)
    p.add_argument("--judge", action="store_true")

    p = sub.add_parser("synthetic", help="generate a QA dataset from live chunks")
    p.add_argument("--n-chunks", type=int, default=10)
    p.add_argument("--questions", type=int, default=3)
    p.add_argument("--dataset-id")

    p = sub.add_parser("compare", help="Wilcoxon test between two runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser("pareto", help="cost-latency-quality frontier over saved runs")
    p.add_argument("--metric", default="faithfulness")
    p.add_argument("--floor", type=float, default=0.7)
    p.add_argument("--dataset")

    args = parser.parse_args()
    setup_logging()

    if args.command == "eval":
        run = ev.run_eval(ev.load_dataset(args.dataset), args.pipeline, args.judge)
        print(f"\nRun {run['run_id']}: {args.pipeline} on {args.dataset}")
        print_table([{"metric": k, "value": v} for k, v in run["summary"].items()], ["metric", "value"])

    elif args.command == "benchmark":
        result = ev.benchmark(ev.load_dataset(args.dataset), args.pipelines, args.judge)
        print_table(result["comparison_table"], ["category", "metric", *result["pipelines"]])
        print("Pareto-optimal (faithfulness ↑, latency ↓, cost ↓):", ", ".join(result["pareto_optimal"]))
        print("Run ids:", result["run_ids"])

    elif args.command == "synthetic":
        dataset = ev.generate_synthetic(args.n_chunks, args.questions, args.dataset_id)
        print(f"Saved datasets/{dataset.dataset_id}.json with {len(dataset.qa_pairs)} QA pairs")

    elif args.command == "compare":
        result = ev.compare_runs(args.baseline, args.candidate, args.alpha)
        print_table(result["rows"], ["metric", "baseline", "candidate", "delta", "p_value", "n", "verdict"])
        print(result["summary"])
        sys.exit(1 if result["has_regression"] else 0)

    elif args.command == "pareto":
        result = ev.pareto(args.metric, args.floor, args.dataset)
        print_table(result["points"], ["run_id", "pipeline", "dataset_id", "quality",
                                       "latency_ms", "cost_usd", "pareto_optimal"])
        print("Recommended:", result["recommended"] or f"nothing on the frontier reaches {args.floor}")


if __name__ == "__main__":
    main()
