"""Stage C — provision the Label Studio project for hand-correction.

Creates a project from ``label_studio_config.xml`` and imports the Stage B
pre-labels so every task opens with the rule extractor's spans already drawn.

Idempotent: re-running against an existing project reuses it and skips the
import if the tasks are already there.

Assumes ``label-studio start`` is already serving. Uses only the standard
library so it can run from any interpreter.

Run:
    python setup_label_studio.py --token <api-token>
    python setup_label_studio.py --token <api-token> --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = NER_DIR / "label_studio_config.xml"
PRELABELS_PATH = NER_DIR / "prelabels" / "prelabels.json"

DEFAULT_URL = "http://localhost:8080"
PROJECT_TITLE = "EstateIQ NER — Stage C hand-correction"
PROJECT_DESCRIPTION = (
    "7-entity NER over real-estate call transcripts. Spans are pre-labeled by the "
    "rule-based extractor (Stage B) and need hand-correction. Negated mentions are "
    "tagged like any other span — see schema.md section 3."
)


class LabelStudioClient:
    """Minimal REST client for the endpoints this setup needs."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(
        self, method: str, path: str, payload: Optional[Any] = None, timeout: int = 300
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Token {self.token}")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else None

    def wait_until_ready(self, attempts: int = 60, delay: float = 2.0) -> bool:
        """Polls until the server answers an authenticated request."""
        for attempt in range(1, attempts + 1):
            try:
                self._request("GET", "/api/projects?page_size=1", timeout=10)
                return True
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise SystemExit(
                        f"Server is up but rejected the API token (HTTP {error.code}). "
                        "Pass the token shown by 'label-studio start'."
                    )
                logger.info("  waiting for server... (attempt %d, HTTP %s)", attempt, error.code)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                logger.info("  waiting for server... (attempt %d)", attempt)
            time.sleep(delay)
        return False

    def find_project(self, title: str) -> Optional[Dict[str, Any]]:
        response = self._request("GET", "/api/projects?page_size=200")
        projects = response.get("results", response) if isinstance(response, dict) else response
        for project in projects or []:
            if project.get("title") == title:
                return project
        return None

    def create_project(self, title: str, label_config: str, description: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/projects",
            {
                "title": title,
                "description": description,
                "label_config": label_config,
                # Keep every task editable in any order rather than a forced queue.
                "show_skip_button": True,
                "enable_empty_annotation": True,
            },
        )

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", f"/api/projects/{project_id}/import", tasks)

    def project(self, project_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/projects/{project_id}")

    def tasks(self, project_id: int, page_size: int = 3) -> Any:
        return self._request(
            "GET", f"/api/tasks?project={project_id}&page_size={page_size}"
        )

    def task_detail(self, task_id: int) -> Dict[str, Any]:
        """The list endpoint omits inline annotation results; this returns them."""
        return self._request("GET", f"/api/tasks/{task_id}")


def load_prelabels() -> List[Dict[str, Any]]:
    with PRELABELS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def report(client: LabelStudioClient, project_id: int, expected_tasks: int) -> bool:
    """Confirms the tasks landed and their pre-label spans survived the import."""
    project = client.project(project_id)
    task_count = project.get("task_number") or 0
    logger.info("")
    logger.info("=" * 74)
    logger.info("IMPORT VERIFICATION")
    logger.info("=" * 74)
    logger.info("Project id            : %s", project_id)
    logger.info("Tasks in project      : %s (expected %s)", task_count, expected_tasks)
    logger.info("Labeling config       : %s", "set" if project.get("label_config") else "MISSING")

    response = client.tasks(project_id, page_size=3)
    listing = response if isinstance(response, dict) else {}
    sample = listing.get("tasks", response if isinstance(response, list) else [])
    annotation_total = listing.get("total_annotations")
    logger.info("Annotations in project: %s (expected %s)", annotation_total, expected_tasks)

    ok = task_count == expected_tasks and annotation_total == expected_tasks
    spans_seen = 0

    # The list endpoint reports counts only; fetch details for the actual spans
    # so this proves the pre-labels really survived the import.
    for stub in (sample or [])[:3]:
        task = client.task_detail(stub["id"])
        annotations = task.get("annotations") or []
        spans = [
            region
            for annotation in annotations
            for region in (annotation.get("result") or [])
        ]
        spans_seen += len(spans)
        logger.info("")
        logger.info(
            "  task %s (%s): %d annotation(s), %d pre-labeled span(s)",
            task.get("id"),
            (task.get("data") or {}).get("transcript_id", "?"),
            len(annotations),
            len(spans),
        )
        raw_text = (task.get("data") or {}).get("text", "")
        for region in spans[:6]:
            value = region.get("value", {})
            start, end = value.get("start"), value.get("end")
            # Re-prove the offsets against the text Label Studio actually stored.
            intact = raw_text[start:end] == value.get("text")
            logger.info(
                "      %-14s [%4s:%4s] %r%s",
                (value.get("labels") or ["?"])[0],
                start,
                end,
                value.get("text"),
                "" if intact else "   <-- OFFSET DRIFT",
            )
            if not intact:
                ok = False

    if task_count != expected_tasks:
        logger.error("Task count does not match the pre-label file.")
    if spans_seen == 0:
        logger.error("No pre-labeled spans came back — annotations did not import.")
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Label Studio base URL")
    parser.add_argument("--token", required=True, help="Label Studio API token")
    parser.add_argument("--title", default=PROJECT_TITLE)
    parser.add_argument(
        "--verify-only", action="store_true", help="Only re-check an existing project."
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    tasks = load_prelabels()
    total_spans = sum(len(t["annotations"][0]["result"]) for t in tasks)
    logger.info("Pre-labels: %d tasks, %d spans from %s", len(tasks), total_spans, PRELABELS_PATH)

    client = LabelStudioClient(args.url, args.token)
    logger.info("Connecting to %s ...", args.url)
    if not client.wait_until_ready():
        raise SystemExit(f"Label Studio never became reachable at {args.url}.")
    logger.info("Server is up.")

    project = client.find_project(args.title)

    if args.verify_only:
        if not project:
            raise SystemExit(f"No project titled {args.title!r}.")
        return 0 if report(client, project["id"], len(tasks)) else 1

    if project:
        logger.info("Reusing existing project %s (%r).", project["id"], args.title)
    else:
        label_config = CONFIG_PATH.read_text(encoding="utf-8")
        project = client.create_project(args.title, label_config, PROJECT_DESCRIPTION)
        logger.info("Created project %s (%r).", project["id"], args.title)

    project_id = project["id"]
    existing = client.project(project_id).get("task_number") or 0
    if existing >= len(tasks):
        logger.info("Project already holds %d tasks — skipping import.", existing)
    else:
        logger.info("Importing %d tasks (this takes a moment)...", len(tasks))
        result = client.import_tasks(project_id, tasks)
        logger.info(
            "Imported: %s tasks, %s annotations.",
            result.get("task_count"),
            result.get("annotation_count"),
        )

    ok = report(client, project_id, len(tasks))

    logger.info("")
    logger.info("=" * 74)
    logger.info("OPEN THIS URL:  %s/projects/%s/data", args.url, project_id)
    logger.info("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
