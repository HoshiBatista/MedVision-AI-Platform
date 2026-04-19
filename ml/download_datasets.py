"""
Download all MedVision datasets from Roboflow.
Reads ROBOFLOW_API_KEY from .env in the project root.

Usage:
    python download_datasets.py            # all datasets
    python download_datasets.py --task mri
    python download_datasets.py --task pneumonia
    python download_datasets.py --task skin
"""

import argparse
import os
import sys
from pathlib import Path

# ── SSL fix for Python 3.13 on macOS (must be before any network import) ─────
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
from roboflow import Roboflow
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root
DATA_ROOT = Path(__file__).parent / "data"   # ml/data/
ENV_PATH = ROOT / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
console = Console()

# ── Dataset registry ──────────────────────────────────────────────────────────
#   Each entry: (workspace, project_slug, version, dest_folder)
DATASETS = {
    "mri": (
        "classification-brain-tumor-teqsl",
        "brain-tumor-segmentation-r4in2",
        2,
        "mri_segmentation",
    ),
    "pneumonia": (
        "siim-cxr",
        "pneumonia-detection-ssibx",
        3,
        "pneumonia_detection",
    ),
    "skin": (
        "skin-nbezt",
        "skin-lesion-jxjgm",
        8,
        "skin_classification",
    ),
}

FORMAT = "yolov11"


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api_key() -> None:
    if not API_KEY:
        console.print(Panel(
            f"[red]ROBOFLOW_API_KEY missing from {ENV_PATH}[/red]\n\n"
            "Add this line to your .env:\n"
            "  [bold]ROBOFLOW_API_KEY=your_key_here[/bold]",
            title="[bold red]Missing API Key[/bold red]",
            expand=False,
        ))
        sys.exit(1)


def count_images(path: Path) -> int:
    return sum(1 for _ in path.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})


def download(task: str) -> None:
    workspace, project_slug, version_num, folder = DATASETS[task]
    dest = DATA_ROOT / folder

    console.print(f"[cyan]Connecting to Roboflow workspace:[/cyan] [bold]{workspace}[/bold]")
    rf = Roboflow(api_key=API_KEY)
    version = rf.workspace(workspace).project(project_slug).version(version_num)

    console.print(f"[cyan]Downloading[/cyan] [bold]{project_slug} v{version_num}[/bold] → [dim]{dest}[/dim]")
    version.download(FORMAT, location=str(dest), overwrite=True)

    n = count_images(dest)
    console.print(f"[green]Done — {n:,} images[/green]\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MedVision datasets from Roboflow")
    parser.add_argument(
        "--task",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to download (default: all)",
    )
    args = parser.parse_args()

    check_api_key()

    # Summary table
    table = Table(title="Datasets to download", show_header=True, header_style="bold blue")
    table.add_column("Task"); table.add_column("Workspace"); table.add_column("Project"); table.add_column("Ver")
    targets = list(DATASETS.keys()) if args.task == "all" else [args.task]
    for t in targets:
        ws, proj, ver, _ = DATASETS[t]
        table.add_row(t, ws, proj, str(ver))
    console.print(table)
    console.print()

    for task in targets:
        console.rule(f"[bold]{task.upper()}[/bold]")
        download(task)

    # Final summary
    console.rule("[bold green]Summary[/bold green]")
    for task in targets:
        _, _, _, folder = DATASETS[task]
        dest = DATA_ROOT / folder
        splits = {s: count_images(dest / s) for s in ("train", "valid", "test") if (dest / s).exists()}
        split_str = "  ".join(f"{k}: {v:,}" for k, v in splits.items())
        console.print(f"[bold]{task:<12}[/bold] {split_str}")

    console.print(f"\n[bold green]All datasets ready.[/bold green] Path: [dim]{DATA_ROOT}[/dim]")
