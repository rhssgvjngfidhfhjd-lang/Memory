"""Create the lightweight isolated Python environments used by baseline adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    "mirix": (
        "demjson3",
        "pathvalidate",
        "pyhumps",
        "colorama",
        "rapidfuzz",
        "Markdown",
        "SpeechRecognition",
        "pydub",
        "google-genai",
        "opentelemetry-instrumentation-requests",
        "opentelemetry-exporter-otlp",
    ),
    "mma": (
        "demjson3",
        "pathvalidate",
        "pyhumps",
        "colorama",
        "rapidfuzz",
        "Markdown",
        "SpeechRecognition",
        "pydub",
        "google-genai",
        "opentelemetry-instrumentation-requests",
        "opentelemetry-exporter-otlp",
    ),
    "memverse": ("pipmaster", "nano-vectordb"),
    "m3_agent": ("matplotlib",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-root",
        default=str(ROOT / ".venvs"),
        help="Environment directory; defaults to <Offline>/.venvs.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=tuple(ENVIRONMENTS),
        default=[],
        help="Create only selected environments; repeat as needed.",
    )
    args = parser.parse_args()
    selected = args.method or list(ENVIRONMENTS)
    env_root = Path(args.env_root).expanduser().absolute()
    for name in selected:
        destination = env_root / name
        if not (destination / "bin" / "python").is_file():
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(destination)
        python = destination / "bin" / "python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *ENVIRONMENTS[name],
            ],
            check=True,
        )
        print(f"ready: {name} -> {python}", flush=True)


if __name__ == "__main__":
    main()
