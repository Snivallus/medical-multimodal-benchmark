"""Minimal OpenAI-compatible chat client for the benchmark pipeline.

The pipeline only needs plain chat completion with a JSON answer, so it talks to
the HTTP endpoint directly instead of depending on an SDK. Configuration comes
from the environment, optionally via ``bench/.env``:

    LLM_API_KEY    falls back to DEEPSEEK_API_KEY / AISTORE_API_KEY
    LLM_BASE_URL   default https://ai.api.coregpu.cn/v1
    LLM_MODEL      default DeepSeek-R1

Any OpenAI-compatible endpoint works (AI STORE, DeepSeek, vLLM, Ollama, ...),
which keeps the pipeline runnable both against a hosted API and a local GPU box.
"""

import json
import os
import re
import time
from pathlib import Path

import requests

ENV_FILE = Path(__file__).resolve().parent / ".env"

DEFAULT_BASE_URL = "https://ai.api.coregpu.cn/v1"
DEFAULT_MODEL = "DeepSeek-R1"

# Retries cover transport errors and throttling. A hung request is the expensive
# case: with four attempts and a long read timeout a single call can block for
# twenty minutes and stall the whole batch, so the budget is bounded here.
DEFAULT_RETRIES = 3
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)

# One pooled session for the whole process; a pipeline run makes thousands of
# calls and reopening a TLS connection for each of them is pure overhead.
_session = requests.Session()


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is available in the environment."""


def load_env(path=ENV_FILE):
    """Read ``KEY=VALUE`` lines from ``.env`` without overriding the environment.

    Keeping this local avoids a dependency on python-dotenv and keeps the key
    out of the repository (``.env`` is git-ignored).
    """

    if not Path(path).exists():
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # A real environment variable always wins over the file.
            os.environ.setdefault(key, value)


load_env()


def api_key():
    """Return the configured API key, or ``None``."""

    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("AISTORE_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )


def api_root():
    """Return the API root, without a trailing slash or endpoint path.

    The AI STORE documentation configures the SDK with the full
    ``.../v1/chat/completions`` endpoint, which is not what an SDK expects. Both
    forms are accepted here so a copied value still works.
    """

    url = (os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]

    return url.rstrip("/")


def model_name():
    """Return the model identifier to request."""

    return os.environ.get("LLM_MODEL") or DEFAULT_MODEL


def is_configured():
    """Return ``True`` when a key is available and calls can be made."""

    return bool(api_key())


def list_models(timeout=(10, 60)):
    """List the models the key can use, newest first if the API says so."""

    key = api_key()

    if not key:
        raise LLMNotConfigured("no API key found; set LLM_API_KEY in bench/.env")

    response = requests.get(
        f"{api_root()}/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()

    return [model.get("id") for model in data.get("data", [])]


def chat(
    messages,
    temperature=0.0,
    max_tokens=1024,
    timeout=(10, 120),
    retries=DEFAULT_RETRIES,
    model=None,
    return_meta=False,
):
    """Send one chat completion and return the assistant text.

    ``model`` overrides the configured model for a single call, which lets the
    mechanical steps run on a fast model and the judgement-heavy steps on a
    stronger one. Transport errors and throttling are retried with an
    exponential backoff; every other 4xx fails immediately because retrying it
    only wastes wall-clock time.

    ``return_meta`` returns ``(text, finish_reason)`` instead of just the text.
    The finish reason distinguishes a model that answered badly from one whose
    answer was cut off at ``max_tokens``, which matters when a thinking model
    shares one budget with its chain of thought.
    """

    key = api_key()

    if not key:
        raise LLMNotConfigured(
            "no API key found; set LLM_API_KEY (or DEEPSEEK_API_KEY) before running"
        )

    payload = {
        "model": model or model_name(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    url = f"{api_root()}/chat/completions"
    last_error = None

    for attempt in range(retries):

        try:
            response = _session.post(url, headers=headers, json=payload, timeout=timeout)

            if response.status_code == 200:
                data = response.json()

                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    finish_reason = choice.get("finish_reason")
                except (AttributeError, KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(f"unexpected response shape: {data}") from error

                if return_meta:
                    return content, finish_reason

                return content

            if response.status_code not in RETRYABLE_STATUS:
                raise RuntimeError(
                    f"LLM request failed with {response.status_code}: {response.text[:300]}"
                )

            last_error = f"HTTP {response.status_code}: {response.text[:120]}"

        except requests.RequestException as error:
            last_error = f"{type(error).__name__}: {error}"

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_error}")


def _scan_json_object(text, start):
    """Return ``(blob, end)`` for the object beginning at ``start``."""

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):

        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start:index + 1], index + 1

    return None, None


def _load_json_object(blob):
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Trailing commas are the most common breakage.
        repaired = re.sub(r",\s*([}\]])", r"\1", blob)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def extract_json(text, last=False):
    """Pull a complete JSON object out of a model answer.

    A greedy ``\\{.*\\}`` match breaks as soon as the answer contains two objects
    or a brace inside a string, so candidates are scanned with brace counting that
    respects string literals.

    ``last=True`` returns the last parseable object. A thinking model often
    echoes the prompt's JSON example inside its reasoning; the real answer is the
    final object, not the first one.
    """

    text = (text or "").replace("```json", "").replace("```", "").strip()
    found = []
    start = text.find("{")

    while start >= 0:
        blob, end = _scan_json_object(text, start)

        if blob is None:
            # An unmatched brace in prose; keep looking for a later object.
            start = text.find("{", start + 1)
            continue

        data = _load_json_object(blob)

        if data is not None:
            found.append(data)
            start = text.find("{", end)
        else:
            start = text.find("{", start + 1)

    if not found:
        return None

    return found[-1] if last else found[0]


def chat_json(messages, model=None, **kwargs):
    """Send one chat completion and parse the answer as a JSON object."""

    text = chat(messages, model=model, **kwargs)
    data = extract_json(text)

    if data is None:
        raise ValueError(f"model did not return JSON: {text[:300]}")

    return data
