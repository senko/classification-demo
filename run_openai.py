"""GPT-6 Luna through the OpenAI Decisions API, POST /v1/decisions (EXPERIMENT.md 6.4).
Never through a chat API. Resumable; results go to $RESULTS_DIR/results.db (default out/)."""

import asyncio
import json
import os

import httpx

from api import (CONCURRENCY, RETRY_STATUSES, TIMEOUT_S, Failed, Retryable, load_config,
                 parse_retry_after, parser, run, select)
from common import CATEGORIES, load_categories, load_env, prompt_input

METHOD = "luna"
BASE_URL = "https://api.openai.com/v1"


def parse(body: dict) -> dict:
    """Prediction fields from a /v1/decisions response body."""
    answers = body.get("answers") or []
    ans = next((a for a in answers if a.get("name") == "category"), answers[0] if answers else None)
    if ans is None:
        return dict(status="unparsable", label="")
    if ans.get("type") == "refusal":
        return dict(status="refusal", label="")
    probs = {p["value"]: float(p["probability"]) for p in ans.get("probabilities") or []}
    if ans.get("type") != "choice" or ans.get("choice") not in CATEGORIES:
        return dict(status="unparsable", label="")
    probs = {c: probs.get(c, 0.0) for c in CATEGORIES}
    return dict(status="ok", label=ans["choice"], probs=probs, confidence=max(probs.values()))


async def main():
    args = parser(__doc__).parse_args()
    load_env()
    cfg = load_config("luna")
    items, prompt, variant = select(args, cfg["main"])
    cats = load_categories()
    question = {
        "name": "category",
        "type": "choice",
        "instructions": cfg["prompts"][prompt]["instructions"],
        "choices": [{"value": c, "description": d} for c, d in zip(cats.id, cats.description)],
    }

    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=TIMEOUT_S,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY),
    ) as client:

        async def call(item) -> dict:
            payload = {"model": cfg["model"], "input": prompt_input(item), "questions": [question]}
            try:
                r = await client.post("/decisions", json=payload)
            except httpx.TransportError as e:
                raise Retryable(f"{type(e).__name__}: {e}") from e
            if r.status_code in RETRY_STATUSES:
                raise Retryable(f"HTTP {r.status_code}: {r.text[:300]}", r.status_code, parse_retry_after(r.headers))
            if r.is_error:
                raise Failed(f"HTTP {r.status_code}: {r.text[:300]}", raw=r.text)
            body = r.json()
            usage = body.get("usage") or {}
            return dict(
                **parse(body), input_tokens=usage.get("input_tokens"),
                cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
                output_tokens=usage.get("output_tokens"),
                reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                model=body.get("model"), provider="openai", raw=json.dumps(body, ensure_ascii=False),
            )

        await run(METHOD, variant, items, call, limit=args.limit, retry_errors=args.retry_errors,
                  concurrency=args.concurrency, notes={"prompt": prompt, "model": cfg["model"]})


if __name__ == "__main__":
    asyncio.run(main())
