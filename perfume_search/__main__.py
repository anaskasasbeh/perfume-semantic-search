"""Command line: python -m perfume_search <command>

build         CSV → embeddings → groups → artifacts/   (needs GEMINI_API_KEY)
serve         run the website + API on http://127.0.0.1:8000
search TEXT   search from the terminal
evaluate      semantic vs keyword evaluation → reports/  (needs GEMINI_API_KEY)
check-images  count working image links → reports/images.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m perfume_search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the artifacts used by the website")
    p_build.add_argument("--raw", type=Path, default=config.RAW_CSV, help="path to the Kaggle CSV")
    p_build.add_argument("--limit", type=int, default=None, help="only use the first N perfumes (quick test)")

    p_serve = sub.add_parser("serve", help="run the site locally")
    p_serve.add_argument("--port", type=int, default=8000)

    p_search = sub.add_parser("search", help="search from the terminal")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=10)

    sub.add_parser("evaluate", help="compare semantic and keyword search")
    sub.add_parser("check-images", help="check product image links")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Arabic text on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        if args.command == "build":
            from .build import build
            from .embeddings import EmbeddingError

            try:
                meta = build(raw_csv=args.raw, limit=args.limit)
            except (EmbeddingError, ValueError) as error:
                logging.error("%s", error)
                return 1
            print(json.dumps(meta, indent=2, ensure_ascii=False))

        elif args.command == "serve":
            import uvicorn

            uvicorn.run("app:app", host="127.0.0.1", port=args.port)

        elif args.command == "search":
            from .engine import SearchEngine

            result = SearchEngine().search(args.query, limit=args.k)
            print(f"mode={result.mode} notice={result.notice} took={result.took_ms:.1f} ms")
            for rank, item in enumerate(result.items, 1):
                notes = ", ".join(item["notes"])
                print(f"{rank:2d}. {item.get('score', 0):.3f}  {item['name']} — {item['brand']}  [{notes}]")

        elif args.command == "evaluate":
            from .evaluate import evaluate, to_markdown

            print(to_markdown(evaluate()))

        elif args.command == "check-images":
            from .images import check_images

            report = check_images()
            print(f"{report['working']} / {report['checked']} image links work ({report['broken']} broken)")

    except FileNotFoundError as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
