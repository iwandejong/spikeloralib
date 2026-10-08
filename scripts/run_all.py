import argparse
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

def _bash(script: Path):
    return ["bash", str(script)]

def _nlu_jobs(out_root: Path):
    return [
        ("V_theta sweep (CoLA)", _bash(SCRIPTS_DIR / "nlu_vtheta_sweep.sh")),
        ("Rank sweep (CoLA)", _bash(SCRIPTS_DIR / "nlu_rank_sweep.sh")),
        ("LR sweep (CoLA)", _bash(SCRIPTS_DIR / "nlu_lr_sweep.sh")),
        ("Full GLUE benchmark", _bash(SCRIPTS_DIR / "nlu_glue_benchmark.sh")),
        ("Sparsity breakdown (CoLA)", _bash(SCRIPTS_DIR / "nlu_sparsity_breakdown.sh")),
        ("Grad-norm comparison (CoLA)", _bash(SCRIPTS_DIR / "nlu_gradnorm_comparison.sh")),
        ("Dropout sweep (CoLA)", _bash(SCRIPTS_DIR / "nlu_dropout_sweep.sh")),
    ]

def _spikegpt_jobs(out_root: Path):
    return [
        ("Subjectivity classification benchmark", _bash(SCRIPTS_DIR / "spikegpt_subj_benchmark.sh")),
    ]

def _llama2_jobs(out_root: Path):
    return [
        ("Llama2-7B QLoRA scaling (CoLA)", _bash(SCRIPTS_DIR / "llama2_scaling_cola.sh")),
    ]

STAGE_BUILDERS = {
    "nlu": ("NLU", _nlu_jobs),
    "spikegpt": ("SpikeGPT-NLU", _spikegpt_jobs),
    "llama2": ("llama2-qlora-scaling", _llama2_jobs),
}

def build_plan(stages, results_root: Path):
    plan = []
    for key in stages:
        subdir, builder = STAGE_BUILDERS[key]
        out_root = results_root / subdir
        for name, train_cmd in builder(out_root):
            plan.append((key, out_root, name, train_cmd))
    return plan

def run_cmd(cmd, env, label):
    print(f"\n{'=' * 88}\n[run_all] {label}\n[run_all] $ {' '.join(cmd)}\n{'=' * 88}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"[run_all] WARNING: exited with code {result.returncode} -- continuing", flush=True)
    return result.returncode == 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=list(STAGE_BUILDERS), default=list(STAGE_BUILDERS))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running anything")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    plan = build_plan(args.stages, results_root)

    if args.dry_run:
        print(f"Plan ({len(plan)} stage(s), results -> {results_root}):")
        for key, out_root, name, train_cmd in plan:
            print(f"  [{key}] {name}")
            print(f"      train: {' '.join(train_cmd)}")
        return 0

    results_root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for key, out_root, name, train_cmd in tqdm(plan, desc="Paper reproduction", unit="stage"):
        out_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["OUT_ROOT"] = str(out_root)
        env.setdefault("PYTHON", sys.executable)

        ok = run_cmd(train_cmd, env, f"[{key}] {name}")
        outcomes.append((key, name, ok))

    print("\n" + "=" * 88)
    print(f"[run_all] Summary ({sum(1 for *_, ok in outcomes if ok)}/{len(outcomes)} training stages OK):")
    for key, name, ok in outcomes:
        print(f"  {'OK  ' if ok else 'FAIL'}  [{key}] {name}")
    print(f"[run_all] Results written under: {results_root} (raw eval_metrics.json per run)")

    return 0 if all(ok for *_, ok in outcomes) else 1

if __name__ == "__main__":
    sys.exit(main())
