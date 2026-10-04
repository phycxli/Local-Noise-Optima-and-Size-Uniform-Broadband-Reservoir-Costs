"""Verify or reproduce the released calculations in a separate working copy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VERIFY = [
    ("audit_final_prl_proofs.py", []),
    ("check_rate_capped_band_optimization.py", []),
    ("check_size_uniform_resources.py", []),
    ("audit_size_uniform_resources.py", []),
    ("check_boundary_argument_certificate.py", []),
]
WORKFLOWS = {
    "verify": VERIFY,
    "figures": [("plot_reservoir_optimization.py", [])],
    "optimize": [
        ("run_rate_capped_band_optimization.py", []),
        ("check_rate_capped_band_optimization.py", []),
        ("run_size_uniform_resources.py", []),
        ("check_size_uniform_resources.py", []),
        ("audit_size_uniform_resources.py", ["--solve-local-construction"]),
        ("plot_reservoir_optimization.py", []),
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest() -> dict:
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT):
            raise ValueError("The release manifest contains an invalid path.")
        if row["path"] in seen:
            raise ValueError("The release manifest contains a duplicate path.")
        seen.add(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Release file changed or missing: {row['path']}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), row["path"], "exec")
    if not seen:
        raise ValueError("The release manifest is empty.")
    return manifest


def prepare_working_copy(output: Path, manifest: dict) -> None:
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("The working output must not contain the release checkout.")
    if output.is_relative_to(ROOT):
        first = output.relative_to(ROOT).parts[0]
        if first != "reproduction" and not first.startswith("reproduction-"):
            raise ValueError("Use reproduction or reproduction-* for output inside the checkout.")
    if (output / ".git").exists():
        raise ValueError("A Git checkout cannot be used as the working output.")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.parts[0] not in ("src", "scripts", "results", "figures"):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or not target.resolve().is_relative_to(output):
            raise ValueError(f"Working output is a symbolic link: {relative}")
        shutil.copyfile(ROOT / relative, target)
    (output / "results/reports").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)


def environment_record() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "cvxpy", "clarabel", "mpmath", "sympy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": versions}


def output_summary(output: Path, workflow: str) -> dict:
    files = ["results/rate_capped_band_optimization_certificate.json",
             "results/size_uniform_resources_certificate.json",
             "results/size_uniform_resources_audit.json",
             "results/final_prl_proof_audit.json",
             "results/boundary_argument_certificate_integrity.json"]
    if workflow in ("figures", "optimize"):
        files += [f"figures/{name}.{extension}" for name in
                  ("fig1_local_device", "fig2_shared_band_cost", "fig3_design_targets")
                  for extension in ("pdf", "png")]
        files.append("results/reports/physical_narrative_figure_audit.json")
    return {relative: sha256(output / relative) for relative in files
            if (output / relative).is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("manifest", *WORKFLOWS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction")
    args = parser.parse_args()
    manifest = check_manifest()
    print(f"Release manifest verified: {len(manifest['files'])} files.", flush=True)
    if args.workflow == "manifest":
        print("PASS: manifest", flush=True)
        return
    output = args.output_dir.resolve()
    prepare_working_copy(output, manifest)
    records = []
    report = {"workflow": args.workflow, "release": manifest["version"],
              "release_manifest_sha256": sha256(ROOT / "release_manifest.json"),
              "environment": environment_record(), "steps": records,
              "published_inputs_preserved": True, "passed": False}
    report_path = output / f"{args.workflow if args.workflow != 'verify' else 'verification'}_report.json"
    for name, arguments in WORKFLOWS[args.workflow]:
        print(f"Running {name} ...", flush=True)
        start = time.monotonic()
        log = output / "logs" / (Path(name).stem + ".txt")
        command = [sys.executable, str(output / "scripts" / name), *arguments]
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=output, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
        record = {"script": f"scripts/{name}", "arguments": arguments,
                  "returncode": completed.returncode,
                  "seconds": round(time.monotonic() - start, 3),
                  "log": log.relative_to(output).as_posix()}
        records.append(record)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if completed.returncode:
            print(f"FAIL: {name}; inspect {log}", file=sys.stderr)
            raise SystemExit(completed.returncode)
        print(f"PASS: {name} ({record['seconds']:.1f}s)", flush=True)
    check_manifest()
    report["output_sha256"] = output_summary(output, args.workflow)
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {args.workflow}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
