"""Haiku and Qwen through OpenRouter chat completions (EXPERIMENT.md 6.5).
Resumable; results go to $RESULTS_DIR/results.db (default out/)."""

import asyncio
import json
import os
from datetime import date

import httpx

from api import (CONCURRENCY, RETRY_STATUSES, TIMEOUT_S, Failed, Retryable, load_config,
                 parse_retry_after, parser, run, select)
from common import CATEGORIES, RESULTS_DIR, load_categories, load_env, prompt_input

BASE_URL = "https://openrouter.ai/api/v1"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "confidence": {"type": "number",
                               "description": "Probability from 0 to 1 that the category is correct"},
            },
            "required": ["category", "confidence"],
            "additionalProperties": False,
        },
    },
}


def system_prompt(template: str) -> str:
    cats = load_categories()
    lines = "\n".join(f"- {c}: {d}" for c, d in zip(cats.id, cats.description))
    return template.strip().replace("{categories}", lines)


def parse(body: dict) -> dict:
    """Prediction fields from a chat completions response body."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    try:
        out = json.loads(content)
        label, confidence = out["category"], float(out["confidence"])
    except (ValueError, TypeError, KeyError):
        label, confidence = None, None
    if label in CATEGORIES:
        return dict(status="ok", label=label, confidence=confidence)
    if message.get("refusal") or choice.get("finish_reason") in ("content_filter", "refusal"):
        return dict(status="refusal", label="")
    return dict(status="unparsable", label="")


def scrub(text: str) -> str:
    """Error body without account fields: OpenRouter error bodies carry the account's user_id."""
    try:
        body = json.loads(text)
    except ValueError:
        return text
    if isinstance(body, dict):
        for key in ["user_id", "workspace_id"]:
            body.pop(key, None)
    return json.dumps(body, ensure_ascii=False)


def usage_fields(usage: dict) -> dict:
    return dict(
        input_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        output_tokens=usage.get("completion_tokens"),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cost_usd=usage.get("cost"),
    )


async def save_pricing(client: httpx.AsyncClient, key: str, model: str):
    """Snapshot of the endpoint price list for this model (EXPERIMENT.md 8.5)."""
    r = await client.get(f"/models/{model}/endpoints")
    r.raise_for_status()
    path = RESULTS_DIR / f"pricing-{date.today().isoformat()}-{key}.json"
    path.write_text(json.dumps(r.json(), indent=1, ensure_ascii=False) + "\n")


async def main():
    ap = parser(__doc__)
    ap.add_argument("model", choices=["haiku", "qwen"])
    args = ap.parse_args()
    load_env()
    cfg = load_config("llm")
    m = cfg["models"][args.model]
    items, prompt, variant = select(args, m["main"])
    p = cfg["prompts"][prompt]
    request = {
        "model": m["model"],
        "provider": m["provider"],
        "response_format": RESPONSE_FORMAT,
        "reasoning": m["reasoning"],
        "max_tokens": cfg["max_tokens"],
        **p.get("request", {}),
    }
    system = system_prompt(p["system"])

    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=TIMEOUT_S,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY),
    ) as client:
        await save_pricing(client, args.model, m["model"])

        async def call(item) -> dict:
            payload = {**request, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_input(item)},
            ]}
            try:
                r = await client.post("/chat/completions", json=payload)
            except httpx.TransportError as e:
                raise Retryable(f"{type(e).__name__}: {e}") from e
            if r.status_code in RETRY_STATUSES:
                text = scrub(r.text)
                raise Retryable(f"HTTP {r.status_code}: {text[:300]}", r.status_code, parse_retry_after(r.headers))
            if r.is_error:
                text = scrub(r.text)
                raise Failed(f"HTTP {r.status_code}: {text[:300]}", raw=text)
            body = r.json()
            # OpenRouter can report an upstream error inside a 200 response.
            if err := body.get("error"):
                code = err.get("code") if isinstance(err, dict) else None
                if code in RETRY_STATUSES:
                    raise Retryable(f"error {code} in body: {str(err)[:300]}", code)
                raise Failed(f"error in body: {str(err)[:300]}", raw=scrub(r.text))
            return dict(
                **parse(body), **usage_fields(body.get("usage") or {}),
                model=body.get("model"), provider=body.get("provider"),
                raw=json.dumps(body, ensure_ascii=False),
            )

        await run(args.model, variant, items, call, limit=args.limit, retry_errors=args.retry_errors,
                  concurrency=args.concurrency, notes={"prompt": prompt, "request": request})


if __name__ == "__main__":
    asyncio.run(main())
