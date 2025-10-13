#!/bin/bash

# Runs mineru benchmark, measuring both olmOCR-bench performance and per document processing performance
#   ./scripts/run_mineru_benchmark.sh
#   ./scripts/run_mineru_benchmark.sh [version]

# Rough steps needed to run:
# uv pip install -U --system "mineru[core]"
# git clone https://huggingface.co/datasets/allenai/olmOCR-bench
# Run mineru -p <input_path> -o <output_path> for each folder in olmOCR-bench/bench_data/pdfs and put all output paths into ~/mineru_bench_[foldername]
# root@triton-cs-aus-455:/build# ls olmOCR-bench/bench_data/pdfs
# arxiv_math  headers_footers  long_tiny_text  multi_column  old_scans  old_scans_math  tables
# #
# mineru -p olmOCR-bench/bench_data/pdfs/arxiv_math -o ~/mineru_bench_arxiv_math
# 
# root@triton-cs-aus-455:/build# ls /root/mineru_bench_arxiv_math/2503.05000_pg9/auto/
# 2503.05000_pg9.md                 2503.05000_pg9_layout.pdf   2503.05000_pg9_model.json  2503.05000_pg9_span.pdf
# 2503.05000_pg9_content_list.json  2503.05000_pg9_middle.json  2503.05000_pg9_origin.pdf  images
#
# So, now, the original pdf was olmOCR-bench/bench_data/pdfs/arxiv_math/2503.05000_pg9.pdf
# you would need to move the mineru_bench_arxiv_math/2503.05000_pg9/auto/2503.05000_pg9.md 
# to olmOCR-bench/bench_data/mineru/arxiv_math/2503.05000_pg9_pg1_repeat1.md 
# Then, once that's all done, run python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data
# For the benchmark job, just run a single mineru convert command and time how long it takes

set -e

# Parse command line arguments
MINERU_VERSION="${1:-latest}"
if [ "$MINERU_VERSION" = "latest" ]; then
    echo "Using latest mineru release"
else
    echo "Using mineru version: $MINERU_VERSION"
fi

# Use conda environment Python if available, otherwise use system Python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "Using conda Python from: $CONDA_PREFIX"
else
    PYTHON="python"
    echo "Warning: No conda environment detected, using system Python"
fi

# Get version from version.py
VERSION=$($PYTHON -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Get current git branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git branch: $GIT_BRANCH"

# Create full image tag
IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Push image to beaker
echo "Trying to push image to Beaker..."
if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
    echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
fi

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_benchmark_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

# Get image tag, beaker user, git branch, git hash, and mineru version from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
mineru_version = sys.argv[5]

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")


# Check if AWS credentials secret exists
aws_creds_secret = f"{beaker_user}-AWS_CREDENTIALS_FILE"
try:
    # Try to get the secret to see if it exists
    b.secret.get(aws_creds_secret, workspace="ai2/olmocr")
    has_aws_creds = True
    print(f"Found AWS credentials secret: {aws_creds_secret}")
except:
    has_aws_creds = False
    print(f"AWS credentials secret not found: {aws_creds_secret}")

# Determine how to install mineru
if mineru_version.lower() == "latest":
    mineru_install_cmd = 'pip install --upgrade "mineru[core]"'
    mineru_version_label = "latest"
else:
    mineru_install_cmd = f'pip install --upgrade "mineru[core]=={mineru_version}"'
    mineru_version_label = mineru_version

setup_mineru_script = """cat <<'PY' > /tmp/run_mineru_cli.py
import shutil
import subprocess
from pathlib import Path

PDF_ROOT = Path("olmOCR-bench/bench_data/pdfs")
TARGET_ROOT = Path("olmOCR-bench/bench_data/mineru")

def find_markdown(source_root, pdf_stem):
    candidates = sorted(source_root.glob(f"**/{pdf_stem}.md"))
    if not candidates:
        raise FileNotFoundError(f"No markdown output found for {pdf_stem}")
    for path in candidates:
        if "auto" in path.parts:
            return path
    return candidates[0]

def main():
    if not PDF_ROOT.exists():
        raise FileNotFoundError(f"PDF root {PDF_ROOT} does not exist")

    if TARGET_ROOT.exists():
        shutil.rmtree(TARGET_ROOT)

    for folder in sorted([p for p in PDF_ROOT.iterdir() if p.is_dir()]):
        output_dir = Path.home() / f"mineru_bench_{folder.name}"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        print(f"Running MinerU on {folder}", flush=True)
        subprocess.run(
            ["mineru", "-p", str(folder), "-o", str(output_dir)],
            check=True,
        )

        pdf_paths = sorted(folder.rglob("*.pdf"))
        if not pdf_paths:
            print(f"No PDFs found in {folder}, skipping")
            continue

        for pdf_path in pdf_paths:
            pdf_stem = pdf_path.stem
            markdown_path = find_markdown(output_dir, pdf_stem)

            relative_parent = pdf_path.parent.relative_to(PDF_ROOT)
            destination_dir = TARGET_ROOT / relative_parent
            destination_dir.mkdir(parents=True, exist_ok=True)

            destination_path = destination_dir / f"{pdf_stem}_pg1_repeat1.md"
            shutil.copyfile(markdown_path, destination_path)
            print(f"Wrote {destination_path}", flush=True)

if __name__ == "__main__":
    main()
PY"""

# First experiment: Original benchmark job
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])
commands.extend([
    "git clone https://huggingface.co/datasets/allenai/olmOCR-bench",
    "cd olmOCR-bench && git lfs pull && cd ..",
    "pip install --upgrade pip",
    mineru_install_cmd,
    setup_mineru_script,
    "python /tmp/run_mineru_cli.py",
    "rm /tmp/run_mineru_cli.py",
    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data --candidate mineru"
])

# Build task spec with optional env vars
task_spec_args = {
    "name": "mineru-benchmark",
    "image": ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),
    "constraints": Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials exist
if has_aws_creds:
    task_spec_args["env_vars"] = [
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret)
    ]

# Create first experiment spec
experiment_spec = ExperimentSpec(
    description=f"MinerU {mineru_version_label} Benchmark Run - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
)

# Create the first experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created benchmark experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
print("-------")
print("")


perf_commands = []
if has_aws_creds:
    perf_commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])
perf_commands.extend([
    "pip install --upgrade pip",
    mineru_install_cmd,
    "pip install awscli",
    "aws s3 cp --recursive s3://ai2-oe-data/jakep/olmocr/olmOCR-mix-0225/benchmark_set/ /root/olmOCR-mix-0225_benchmark_set/",
    "rm -rf /root/olmOCR-mix-0225_benchmark_set_mineru",
    "time mineru -p /root/olmOCR-mix-0225_benchmark_set/ -o /root/olmOCR-mix-0225_benchmark_set_mineru"
])

# Build performance task spec
perf_task_spec_args = {
    "name": "mineru-performance",
    "image": ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    "command": [
        "bash", "-c",
        " && ".join(perf_commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),
    "constraints": Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials exist
if has_aws_creds:
    perf_task_spec_args["env_vars"] = [
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret)
    ]

# Create performance experiment spec
perf_experiment_spec = ExperimentSpec(
    description=f"MinerU {mineru_version_label} Performance Test - Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**perf_task_spec_args)],
)

# Create the performance experiment
perf_experiment = b.experiment.create(spec=perf_experiment_spec, workspace="ai2/olmocr")
print(f"Created performance experiment: {perf_experiment.id}")
print(f"View at: https://beaker.org/ex/{perf_experiment.id}")
EOF

# Run the Python script to create the experiments
echo "Creating Beaker experiments..."
$PYTHON /tmp/run_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH $MINERU_VERSION

# Clean up temporary file
rm /tmp/run_benchmark_experiment.py

echo "Benchmark experiments submitted successfully!"
