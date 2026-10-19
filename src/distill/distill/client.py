"""One HTTP client for every model — local vLLM, NIM, the NVIDIA gateway, OpenAI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import requests

from .parsing import THINK_RE, LabelResult, error_result, parse
from .prompts import BlankInput, PromptSet
from .schema import early_stop_sequences, response_format

logger = logging.getLogger(__name__)

_NEW_OPENAI_FAMILIES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def uses_new_openai_params(model: str) -> bool:
    """gpt-5/o-series take `max_completion_tokens` and reject a custom temperature."""
    lowered = model.lower()
    return any(family in lowered for family in _NEW_OPENAI_FAMILIES)


def resolve_token_param(model: str, override: str | None = None) -> str:
    return override or ("max_completion_tokens" if uses_new_openai_params(model) else "max_tokens")


def extract_assistant_content(data: dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        message = first

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        combined = "".join(parts).strip()
        if combined:
            return combined
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def finish_reason(data: dict[str, Any]) -> str | None:
    """`length` means the model ran out of budget mid-answer."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


@dataclass
class EndpointConfig:
    url: str
    model: str
    api_key: str = "EMPTY"
    workers: int = 24
    max_output_tokens: int = 3000
    timeout: int = 300
    retries: int = 3
    temperature: float = 0.0
    send_temperature: bool | None = None
    token_param: str | None = None
    guided_json: bool = False
    early_stop: bool = False
    extra_body: dict[str, Any] = None         # type: ignore[assignment]
    health_attempts: int = 20
    health_delay: float = 15.0
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None

    def __post_init__(self) -> None:
        if self.extra_body is None:
            self.extra_body = {}


def base_url(url: str) -> str:
    """The server root behind a chat-completions URL: everything before `/v1/`."""
    return url.split("/v1/")[0].rstrip("/")


def wait_for_url(url: str, *, attempts: int, delay: float, verbose: bool = True) -> None:
    """Bounded readiness gate, as a function.

    Module level rather than a method because `serving.serve_model` needs the same gate and a
    second implementation of it would drift: this one is what every stage has waited on.
    """
    if attempts <= 0:
        if verbose:
            print("health gate: skipped")
        return
    base = base_url(url)
    probes = [f"{base}/health", f"{base}/v1/models"]
    for attempt in range(1, attempts + 1):
        for probe in probes:
            try:
                if requests.get(probe, timeout=5).status_code == 200:
                    if verbose:
                        print(f"health gate: {probe} -> 200 (attempt {attempt})")
                    return
            except requests.RequestException:
                pass
        if attempt < attempts:
            if verbose:
                print(f"health gate: not ready ({attempt}/{attempts}), retrying in "
                      f"{delay:.0f}s", flush=True)
            time.sleep(delay)
    raise SystemExit(
        f"endpoint not ready after {attempts} probes of {probes[0]} — is the server up? "
        f"Print the recipe with `distill serve-cmd`, start it with -d and WITHOUT --rm so a "
        f"crash leaves logs, and check `du -sh` on the merged model first. "
        f"Set health.attempts: 0 in the config to skip this gate."
    )


def endpoint_names(url: str, *, timeout: float = 5.0) -> list[str] | None:
    """What the server says it serves, from `/v1/models`. None when it cannot be asked.

    This is the only thing on the wire that names the weights, and it names them loosely -- it
    is the `--served-model-name` the container was started with, not the checkpoint. So it
    catches a container left over from another role or another granularity, and cannot catch a
    right name in front of wrong weights. See `serving.granularity_evidence` for that half.
    """
    try:
        response = requests.get(f"{base_url(url)}/v1/models", timeout=timeout)
        if response.status_code != 200:
            return None
        return [entry.get("id", "") for entry in response.json().get("data", [])]
    except (requests.RequestException, ValueError):
        return None


class LabelClient:
    """Ask one endpoint for one row's label, through one prompt set."""

    def __init__(self, prompts: PromptSet, endpoint: EndpointConfig, *, harvest: bool = False) -> None:
        self.prompts = prompts
        self.endpoint = endpoint
        self.harvest = harvest
        self.token_param = resolve_token_param(endpoint.model, endpoint.token_param)
        send = endpoint.send_temperature
        self._send_temperature = (not uses_new_openai_params(endpoint.model)) if send is None else send
        self._temperature_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        if endpoint.guided_json and (endpoint.extra_body.get("chat_template_kwargs") or {}).get(
            "enable_thinking"
        ):
            logger.warning(
                "guided_json with reasoning enabled: schema-constrained decoding leaves no room "
                "for a reasoning trace. Expect empty or truncated traces."
            )

    def generation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {self.token_param: self.endpoint.max_output_tokens}
        if self._send_temperature:
            params["temperature"] = self.endpoint.temperature
        if self.endpoint.guided_json:
            params["response_format"] = response_format(self.prompts)
        if self.endpoint.early_stop:
            stops = early_stop_sequences(self.prompts)
            if stops:
                params["stop"] = stops
        params.update(self.endpoint.extra_body)
        return params

    def build_payload(self, row: Mapping[str, Any], content_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": self.prompts.messages(row, content_type),
        }
        payload.update(self.generation_params())
        return payload

    def _record_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        with self._usage_lock:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.usage["calls"] += 1

    def _post(self, payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
        """(content, finish_reason, transport_error)."""
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.endpoint.api_key}"}
        last_error: str | None = None
        for attempt in range(1, self.endpoint.retries + 1):
            try:
                response = requests.post(self.endpoint.url, headers=headers, json=payload,
                                         timeout=self.endpoint.timeout)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    return "", None, "malformed response body (not a JSON object)"
                self._record_usage(data)
                return extract_assistant_content(data), finish_reason(data), None
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                detail = ""
                if exc.response is not None:
                    detail = (exc.response.text or "")[:300].replace("\n", " ")
                last_error = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
                if status == 400 and self._drop_temperature(detail, payload):
                    continue
                if status in (429, 502, 503) and attempt < self.endpoint.retries:
                    time.sleep(2 ** attempt)
                    continue
                return "", None, last_error
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.endpoint.retries:
                    time.sleep(2 ** attempt)
                    continue
                return "", None, last_error
        return "", None, last_error

    def _drop_temperature(self, detail: str, payload: dict[str, Any]) -> bool:
        """Handle a model that refuses a temperature override — once, for the whole run."""
        if "temperature" not in detail.lower() or "temperature" not in payload:
            return False
        with self._temperature_lock:
            if self._send_temperature:
                self._send_temperature = False
                logger.warning(
                    "%s rejected `temperature`; dropping it for the rest of the run "
                    "(server said: %s)", self.endpoint.model, detail[:160],
                )
        payload.pop("temperature", None)
        return True

    def label(self, row: Mapping[str, Any], content_type: str) -> LabelResult:
        started = time.monotonic()
        try:
            payload = self.build_payload(row, content_type)
        except BlankInput as exc:
            return error_result(str(exc), kind="schema")
        except (KeyError, ValueError) as exc:
            return error_result(f"cannot build the prompt: {exc}", kind="schema")

        content, finish, transport_error = self._post(payload)
        if transport_error is not None:
            result = error_result(transport_error, kind="http")
        else:
            result = parse(content, self.prompts)
            if not result.ok and finish == "length":
                result.error = (f"truncated at the token budget "
                                f"({self.endpoint.max_output_tokens}). {result.error}")
                result.error_kind = "budget"
            if not result.ok and self.harvest and content:
                harvested = self._harvest(payload["messages"], content)
                if harvested is not None:
                    harvested.extra["harvested"] = True
                    harvested.extra["first_error"] = result.error
                    harvested.reasoning_trace = content
                    result = harvested
        result.extra["latency_s"] = round(time.monotonic() - started, 3)
        result.extra.setdefault("finish_reason", finish)
        if not result.ok:
            result.extra.setdefault("raw_on_failure", (content or "")[-600:])
        return result

    def _harvest(self, messages: list[dict[str, str]], partial: str) -> LabelResult | None:
        """Second pass: hand back the unfinished reasoning, ask only for the answer."""
        fields = ", ".join(f'"{f}"' for f in self.prompts.output_fields())
        followup = [
            *messages,
            {"role": "assistant", "content": partial},
            {"role": "user", "content":
                "Your analysis was cut off before you gave an answer. Do not analyse further. "
                f"Reply with ONLY the JSON object ({fields}), based on the reasoning above."},
        ]
        payload: dict[str, Any] = {"model": self.endpoint.model, "messages": followup,
                                  self.token_param: 160,
                                  "response_format": response_format(self.prompts)}
        if self._send_temperature:
            payload["temperature"] = self.endpoint.temperature
        extra = dict(self.endpoint.extra_body)
        template_kwargs = dict(extra.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        extra["chat_template_kwargs"] = template_kwargs
        payload.update(extra)

        content, finish, transport_error = self._post(payload)
        if transport_error is not None or not content:
            return None
        harvested = parse(content, self.prompts)
        harvested.extra["finish_reason"] = finish
        return harvested if harvested.ok else None

    def stray_reasoning(self, row: Mapping[str, Any], content_type: str
                        ) -> tuple[str, str] | None:
        """One real call, to see whether the model writes a reasoning trace it must not.

        A student is trained to emit `<think></think>` and then the JSON, and is served with a
        chat template whose `enable_thinking` defaults to False. So a NON-EMPTY `<think>` block
        is not a stylistic difference: it means the template actually loaded is the base
        model's, which opens one on `add_generation_prompt` whatever `enable_thinking` says.
        Serving that way was measured at 2.26x fewer calls/s and accuracy 94% -> 33.5%, with
        nothing in the output saying so. Passing `chat_template_kwargs: {enable_thinking: false}`
        does NOT protect against it -- the base template ignores the flag, and our own template
        already defaults to off, so the flag is redundant where it works and useless where it
        would be needed. This is the check that actually sees the fault.

        Only meaningful for a student. A teacher reasons on purpose: turning it off cost
        92.8% -> 62.7%, so `prompt_style == "full"` returns None without spending a call.

        Returns ("error", message) for a `<think>` block with content -- deterministic, the
        template does it on every row -- ("warn", message) for prose before the JSON, which
        may be one row misbehaving, and None when the answer is clean.
        """
        if self.prompts.prompt_style != "stub":
            return None
        result = self.label(row, content_type)
        raw = result.raw_response or ""
        thought = " ".join(t[7:-8].strip() for t in THINK_RE.findall(raw)).strip()
        if not thought and "<think>" in raw:
            thought = raw.split("<think>", 1)[1].split("</think>")[0].strip()
        if thought:
            return ("error", (
                f"the student answered with a non-empty <think> block, which it was never "
                f"trained to produce:\n    {thought[:200]!r}\n"
                f"The chat template loaded by the server is the base model's, not "
                f"`ops/n34b_nothink_template.jinja`. Restart it with --chat-template (the line "
                f"`distill --run <run> serve-cmd student` prints already carries the flag). "
                f"Measured cost of labelling this way: accuracy 94% -> 33.5%, silently.\n"
                f"`chat_template_kwargs: {{enable_thinking: false}}` will NOT fix this."
            ))
        if result.reasoning_trace.strip():
            return ("warn", (
                f"the student wrote prose before the JSON: "
                f"{result.reasoning_trace.strip()[:160]!r}. No <think> block, so the template "
                f"is probably right and this may be one row -- but a student served with the "
                f"stub should answer with the JSON and nothing else."
            ))
        return None

    def require_no_stray_reasoning(self, row: Mapping[str, Any], content_type: str) -> None:
        """`stray_reasoning`, as a gate. One call, before a metered pass writes anything."""
        verdict = self.stray_reasoning(row, content_type)
        if verdict is None:
            return
        severity, message = verdict
        if severity == "error":
            raise SystemExit("template probe: " + message)
        print("WARNING  template probe: " + message)

    def wait_for_endpoint(self, *, verbose: bool = True) -> None:
        """Bounded readiness gate."""
        wait_for_url(self.endpoint.url, attempts=self.endpoint.health_attempts,
                     delay=self.endpoint.health_delay, verbose=verbose)

    def cost(self) -> dict[str, Any] | None:
        """What this run cost, from the API's own usage field rather than an estimate."""
        if self.endpoint.price_in_per_m is None or self.endpoint.price_out_per_m is None:
            return None
        with self._usage_lock:
            prompt = self.usage["prompt_tokens"]
            completion = self.usage["completion_tokens"]
            calls = self.usage["calls"]
        total = (prompt * self.endpoint.price_in_per_m + completion * self.endpoint.price_out_per_m) / 1e6
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "calls_with_usage": calls,
            "usd": round(total, 4),
            "usd_per_call": (total / calls) if calls else None,
        }
