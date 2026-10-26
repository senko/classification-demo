"""Jev through the TypeSafe API (EXPERIMENT.md 6.3).
Resumable; results go to $RESULTS_DIR/results.db (default out/)."""

import asyncio

from typesafe_sdk import (AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeAPIConnectionError,
                          TypeSafeAPIError)

from api import (RETRY_STATUSES, TIMEOUT_S, Failed, Retryable, load_config, parse_retry_after,
                 parser, run, select)
from common import CATEGORIES, jev_state, load_categories, load_env

METHOD = "jev"


def question(prompt: dict) -> Choice:
    cats = load_categories()
    descriptions = dict(zip(cats.id, cats.description))
    match prompt["criteria"]:
        case "string":
            criteria = descriptions
        case "object":
            criteria = {c: {"covers": d} for c, d in descriptions.items()}
        case other:
            raise SystemExit(f"unknown criteria form {other!r}")
    return Choice(instructions=prompt["instructions"], criteria=criteria)


async def main():
    args = parser(__doc__).parse_args()
    load_env()
    cfg = load_config("jev")
    items, prompt, variant = select(args, cfg["main"])
    questions = {"category": question(cfg["prompts"][prompt])}

    # SDK retries off: the shared policy in api.py retries, so attempt counts match the other methods.
    async with AsyncTypeSafeClient(model=cfg["model"], timeout=TIMEOUT_S,
                                   retry=RetryPolicy(max_retries=0)) as client:

        async def call(item) -> dict:
            try:
                resp = await client.system_one(state=jev_state(item), questions=questions)
            except TypeSafeAPIConnectionError as e:
                raise Retryable(f"{type(e).__name__}: {e}") from e
            except TypeSafeAPIError as e:
                if e.status in RETRY_STATUSES:
                    raise Retryable(str(e), e.status, parse_retry_after(e.headers)) from e
                raise Failed(str(e), raw=None if e.body is None else str(e.body)) from e
            raw = resp.raw_http_response.text
            ans = resp.choices.get("category")
            if ans is None or ans.choice not in CATEGORIES:
                return dict(status="unparsable", label="", raw=raw, model=resp.model, provider="typesafe",
                            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
            probs = {c: float(ans.probabilities.get(c, 0.0)) for c in CATEGORIES}
            return dict(
                status="ok", label=ans.choice, probs=probs, confidence=max(probs.values()),
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                model=resp.model, provider="typesafe", raw=raw,
            )

        await run(METHOD, variant, items, call, limit=args.limit, retry_errors=args.retry_errors,
                  concurrency=args.concurrency, notes={"prompt": prompt, "model": cfg["model"]})


if __name__ == "__main__":
    asyncio.run(main())
