"""EstateIQ Flask entrypoint.

Run:
    python -m app
    python app.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routes import webhook_bp  # noqa: E402

logger = logging.getLogger(__name__)


def create_app(preload_models: bool = True) -> Flask:
    """Builds the Flask app.

    ``preload_models`` loads the NER model during startup rather than on the
    first webhook. A cold load takes seconds, and Vapi's webhook has a timeout,
    so the first real call should not be the one paying that cost. Tests pass
    ``False`` to stay fast.
    """
    app = Flask(__name__)
    app.register_blueprint(webhook_bp)

    @app.route("/")
    def home():
        return "EstateIQ Flask Server Running"

    @app.route("/health")
    def health():
        from nlp import predict

        return jsonify({
            "status": "ok",
            "extractor_loaded": predict._INSTANCE is not None,
        })

    if preload_models:
        from nlp.predict import get_hybrid_extractor

        # Loading here rather than on first request matters for the live demo:
        # a cold load is several seconds, and that must not happen mid-call.
        logger.info("Loading extraction model (this takes a few seconds)...")
        started = time.time()
        get_hybrid_extractor()
        banner = "=" * 60
        print(banner, flush=True)
        print(f"  ESTATEIQ IS READY  (model loaded in {time.time() - started:.1f}s)",
              flush=True)
        print("  Webhook: POST /vapi/webhook", flush=True)
        print(banner, flush=True)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("FLASK_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Load the NER model lazily on first request instead of at startup.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = create_app(preload_models=not args.no_preload)
    # debug=True spawns a reloader that would load the model twice.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
