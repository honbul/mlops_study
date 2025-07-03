#!/usr/bin/env python3
"""
wrapper_mlflow.py
────────────────────────────────────────
* timm 학습 로그를 실시간으로 MLflow 에 올림
* 매 에폭마다 summary.csv 를 파싱해 metric 기록
* 새 checkpoint-*.pth.tar 가 생길 때마다 artifact 로 업로드
"""

import argparse, subprocess, pathlib, csv, time, mlflow, os, sys, re, zipfile

# ────────── CLI
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)          # 예: /output
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=64)
    return p.parse_args()

# ────────── metric
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")
def push_metrics(csv_path: pathlib.Path, seen_epochs: set[int]):
    """
    summary.csv 에서 아직 업로드하지 않은 epoch 행을 찾아 MLflow log
    """
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)

            metrics = {
                k: float(v) for k, v in row.items()
                if k != "epoch" and v and _FLOAT_RE.fullmatch(v)
            }
            if metrics:
                mlflow.log_metrics(metrics, step=epoch)
                print(f"[metric] epoch {epoch}", flush=True)

# ────────── artifact
def log_artifact(path: pathlib.Path, subdir: str | None = None, seen: set[str] | None = None):
    """
    path 가 아직 업로드되지 않았으면 MLflow artifact 에 기록
    """
    name = path.name
    if seen is not None and name in seen:
        return
    mlflow.log_artifact(str(path), artifact_path=subdir)
    print(f"[artifact] {subdir or '.'}/{name}", flush=True)
    if seen is not None:
        seen.add(name)

# ────────── timm 도움말에서 플래그 지원 여부 확인
def timm_has(flag: str) -> bool:
    try:
        txt = subprocess.check_output(
            ["python", "/workspace/timm/train.py", "-h"],
            text=True, stderr=subprocess.DEVNULL
        )
        return flag in txt
    except subprocess.CalledProcessError:
        return False

# ────────── main
def main():
    args = get_args()

    # MLflow 기본 설정
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "timm"))

    run_name = f"{args.model}_{int(time.time())}"
    run_root = pathlib.Path(args.output)
    run_dir  = run_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    # timm train.py 실행 인자
    cmd = [
        "python", "/workspace/timm/train.py",
        "--model", args.model,
        "--data-dir", args.data_dir,
        "--output", args.output,
        "--experiment", run_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
    ]
    if timm_has("--checkpoint-hist"):
        cmd += ["--checkpoint-hist", "3"]
    if timm_has("--save-every"):
        cmd += ["--save-every", "1"]
    if timm_has("--no-clean"):
        cmd += ["--no-clean"]       # 삭제 방지 옵션이 있으면 사용

    print("▶", " ".join(cmd), flush=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        # timm 학습 프로세스 실행
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        seen_epochs: set[int] = set()
        seen_ckpt:   set[str] = set()

        while proc.poll() is None:
            # 1) timm stdout 중계
            line = proc.stdout.readline()
            if line:
                print("[timm]", line.rstrip(), flush=True)

            # 2) metric 실시간 업로드
            csv_file = run_dir / "summary.csv"
            if csv_file.exists():
                push_metrics(csv_file, seen_epochs)

            # 3) 새 checkpoint 발견 시 업로드
            for ckpt in run_dir.glob("checkpoint-*.pth.tar"):
                log_artifact(ckpt, subdir="checkpoints", seen=seen_ckpt)

            time.sleep(2)

        # ───── 학습 종료 후 남은 것 정리 ─────
        if (run_dir / "summary.csv").exists():
            push_metrics(run_dir / "summary.csv", seen_epochs)
            log_artifact(run_dir / "summary.csv", subdir="logs")
        if (run_dir / "args.yaml").exists():
            log_artifact(run_dir / "args.yaml", subdir="logs")

        # (선택) ZIP 파일(≤200 MB) 업로드
        zip_file = run_dir.with_suffix(".zip")
        if zip_file.exists() and zip_file.stat().st_size <= 200 * 2**20:
            log_artifact(zip_file, subdir="run_zip")

        # timm 프로세스 반환 코드 확인
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

# ──────────
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[wrapper ERROR] {e}", file=sys.stderr, flush=True)
        sys.exit(1)
