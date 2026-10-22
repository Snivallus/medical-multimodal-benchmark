"""Check that the configured LLM endpoint is reachable and usable.

Run this first after filling in ``bench/.env``:

    python bench/check_api.py            # list models, then send one tiny prompt
    python bench/check_api.py --models   # only list models

The model ids printed by ``--models`` are the values to put in ``LLM_MODEL``.
"""

import argparse
import sys

import llm


def main():
    parser = argparse.ArgumentParser(description="Verify LLM credentials.")
    parser.add_argument("--models", action="store_true", help="only list the models")
    parser.add_argument("--prompt", default="回复两个字：收到", help="test prompt")
    args = parser.parse_args()

    print(f"endpoint: {llm.api_root()}")
    print(f"model:    {llm.model_name()}")
    print(f"api key:  {'set' if llm.is_configured() else 'MISSING'}")

    if not llm.is_configured():
        print("\nadd LLM_API_KEY to bench/.env (see bench/.env.example) and retry")
        return 1

    try:
        models = llm.list_models()
    except Exception as error:
        print(f"\ncould not list models: {error}")
        models = []

    if models:
        print(f"\nmodels available ({len(models)}):")
        for name in models:
            mark = " <- configured" if name == llm.model_name() else ""
            print(f"  {name}{mark}")

    if args.models:
        return 0

    print("\nsending a test prompt ...")

    try:
        answer = llm.chat([{"role": "user", "content": args.prompt}], max_tokens=32)
    except Exception as error:
        print(f"chat failed: {error}")
        return 1

    print(f"reply: {answer.strip()[:200]}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
