"""EstateIQ NLP demo driver.

Sends prepared call transcripts to the running webhook and prints the raw
conversation next to the structured features extracted from it, so the
before/after is obvious to someone watching.

Standard library only, ASCII output only - nothing here can fail on a lecture
room machine because of a missing package or a console codepage.

Run:
    python demo/run_demo.py              # all four, pausing between each
    python demo/run_demo.py --no-pause   # straight through
    python demo/run_demo.py --only 3     # one case
    python demo/run_demo.py --check      # readiness check only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DEMO_DIR = Path(__file__).resolve().parent
TRANSCRIPTS = DEMO_DIR / "transcripts.json"
DEFAULT_URL = "http://localhost:5000"
WIDTH = 78

FIELD_ORDER = [
    "city", "area", "property_type", "category", "bhk",
    "budget", "budget_value", "furnishing", "amenities", "negated",
]


def rule(char: str = "=") -> str:
    return char * WIDTH


def build_payload(case: Dict[str, Any]) -> Dict[str, Any]:
    """Wraps the turns in the Vapi end-of-call-report shape."""
    return {
        "message": {
            "type": "end-of-call-report",
            "call": {"id": f"demo_call_{case['id']}"},
            "customer": {"number": "+919876543210"},
            "endedReason": "customer-ended-call",
            "artifact": {
                "messages": [
                    {"role": role, "message": text} for role, text in case["turns"]
                ]
            },
        }
    }


def post(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/vapi/webhook",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_ready(url: str) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return None


def check(url: str) -> bool:
    health = server_ready(url)
    print()
    if health is None:
        print("  [X] SERVER NOT RUNNING at " + url)
        print()
        print("      Start it first, in a separate terminal:")
        print("          cd backend")
        print("          python app.py")
        print()
        print("      Wait for:  EstateIQ IS READY")
        return False
    loaded = health.get("extractor_loaded")
    print("  [OK] Server is up at " + url)
    print("  [OK] Extraction model loaded: " + str(loaded))
    if not loaded:
        print()
        print("      NOTE: model not preloaded - the first case will be slow.")
        print("      Restart with 'python app.py' (preload is on by default).")
    print()
    return True


def print_case(case: Dict[str, Any], result: Dict[str, Any], elapsed: float) -> None:
    print()
    print(rule())
    print(f"  DEMO {case['id']}/4  -  {case['title'].upper()}")
    print(f"  Shows: {case['shows']}")
    print(rule())
    print()
    print("  THE CALL (what the customer actually said)")
    print("  " + rule("-")[:WIDTH - 2])
    for line in (result.get("transcript") or "").splitlines():
        speaker, _, said = line.partition(": ")
        print(f"    {speaker + ':':<10} {said}")
    print()
    print("  EXTRACTED FEATURES (structured, ready for scoring)")
    print("  " + rule("-")[:WIDTH - 2])

    features = result.get("features") or {}
    ordered = {key: features[key] for key in FIELD_ORDER if key in features}
    for key in features:
        ordered.setdefault(key, features[key])

    for key, value in ordered.items():
        if value in (None, [], ""):
            shown = "-"
        elif isinstance(value, list):
            shown = ", ".join(str(v) for v in value)
        elif key == "budget_value":
            shown = f"{value:,.0f} rupees"
        else:
            shown = str(value)
        marker = " "
        if key == "negated" and value:
            marker = "!"
        print(f"    {marker} {key:<15} {shown}")

    filled = sum(
        1 for k, v in features.items()
        if k not in ("negated",) and v not in (None, [], "")
    )
    print()
    print(f"  -> {filled} fields extracted in {elapsed:.2f}s")
    print()
    print("  POINT OUT: " + case["point_out"])
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--only", type=int, help="Run a single demo by id (1-4).")
    parser.add_argument("--no-pause", action="store_true",
                        help="Do not wait for Enter between cases.")
    parser.add_argument("--check", action="store_true",
                        help="Only verify the server is ready.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    print(rule())
    print("  ESTATEIQ  -  NLP FEATURE EXTRACTION DEMO")
    print("  Voice transcript  ->  structured lead features")
    print(rule())

    if not check(args.url):
        return 1
    if args.check:
        return 0

    cases: List[Dict[str, Any]] = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"  No demo with id {args.only}.")
            return 1

    for index, case in enumerate(cases):
        if not args.no_pause:
            try:
                input(f"  [Press Enter for demo {case['id']}: {case['title']}] ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

        started = time.time()
        try:
            result = post(args.url, build_payload(case))
        except Exception as error:  # noqa: BLE001 - demo must fail readably
            print()
            print(f"  [X] Request failed: {type(error).__name__}: {error}")
            print("      Is the server still running? Re-run with --check.")
            return 1
        elapsed = time.time() - started

        if result.get("status") != "processed":
            print(f"  [X] Unexpected response: {json.dumps(result)[:300]}")
            return 1

        print_case(case, result, elapsed)
        if index < len(cases) - 1:
            print(rule("-"))

    print(rule())
    print("  DEMO COMPLETE  -  all cases extracted successfully.")
    print(rule())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
