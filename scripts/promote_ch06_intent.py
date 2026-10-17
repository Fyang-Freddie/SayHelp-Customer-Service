"""Recheck both frozen Jev splits and promote the intent policy only if safe."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from app.config import Settings
from app.evaluation.ch06 import fingerprints
from app.evaluation.intent_recalibration import promote_verified_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.env_file:
        for name, value in dotenv_values(args.env_file).items():
            if value is not None:
                os.environ.setdefault(name, value)
    result = promote_verified_candidate(ROOT, fingerprints(Settings.from_env()))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
