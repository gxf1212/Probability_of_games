"""Command-line pipeline to run the Changchun Mahjong egg simulation."""
from __future__ import annotations

import argparse
from pathlib import Path

from pathlib import Path

from .metrics import compute_summary, save_summary
from .plotting import (
    plot_combo_distribution,
    plot_convergence,
    plot_egg_pie,
    plot_purity_bars,
)
from .report import FigureBundle, build_report
from .samples import HandSamples, generate_samples, load_samples, save_samples
from .simulation import SimulationConfig, run_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "report.md"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
RESULTS_PATH = OUTPUT_DIR / "results.parquet"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_PATH = DATA_DIR / "hands.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Changchun Mahjong egg simulator")
    parser.add_argument("--trials", type=int, default=100_000, help="Number of Monte Carlo samples when generating")
    parser.add_argument("--seed", type=int, default=2025_1204, help="Random seed for sampling")
    parser.add_argument("--samples-path", type=Path, default=SAMPLES_PATH, help="Parquet file storing hand samples")
    parser.add_argument("--regen-samples", action="store_true", help="Force regenerate samples even if file exists")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH, help="Parquet file for simulation outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(trials=args.trials, seed=args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    samples: HandSamples
    if args.samples_path.exists() and not args.regen_samples:
        samples = load_samples(args.samples_path)
        print(f"Loaded {samples.trials:,} samples from {args.samples_path}")
    else:
        samples = generate_samples(args.trials, seed=args.seed)
        save_samples(samples, args.samples_path)
        print(f"Generated and saved {samples.trials:,} samples to {args.samples_path}")

    config = SimulationConfig(trials=samples.trials, seed=args.seed)
    result = run_simulation(config, samples=samples)

    pie_path = FIGURES_DIR / "egg_probabilities.png"
    combo_path = FIGURES_DIR / "egg_combo_distribution.png"
    convergence_path = FIGURES_DIR / "convergence.png"

    summary = compute_summary(result.frame, config=config)
    save_summary(summary, SUMMARY_PATH)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(args.results_path, index=False)

    plot_egg_pie(summary, pie_path)
    plot_combo_distribution(summary, combo_path)
    plot_convergence(summary, convergence_path)
    purity_path = FIGURES_DIR / "egg_purity.png"
    plot_purity_bars(summary, purity_path)

    figures = FigureBundle(
        pie=pie_path,
        combo_distribution=combo_path,
        convergence=convergence_path,
        purity=purity_path,
    )
    report_text = build_report(summary, figures, REPORT_PATH.parent, trials=config.trials, seed=config.seed)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
