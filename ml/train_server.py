import os
import sys
import json
import base64
import shutil
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml
from flask import Flask, request, jsonify, send_file, Response
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
# Canonical subdirectories of ml/ (kept as placeholders via .gitkeep; contents are gitignored).
WEIGHTS_DIR = BASE_DIR / "weights"       # base pretrained checkpoints (yolo11n.pt etc.)
UPLOADS_DIR = BASE_DIR / "uploads"     # raw uploaded videos
EXTRACTIONS_DIR = BASE_DIR / "extractions"  # extracted raw frame JPGs (original frames per run
DATASETS_DIR = BASE_DIR / "datasets"  # train/val image + label splits + dataset.yaml per run
OUTPUTS_DIR = BASE_DIR / "outputs"   # training run folders and final .pt exports
TMP_DIR = BASE_DIR / "_tmp"             # transient temp files (e.g. shutil rmtree helpers etc.)
TEST_DIR = BASE_DIR / "test"            # user-provided real-world images → post-train smoke test

for d in [WEIGHTS_DIR, UPLOADS_DIR, EXTRACTIONS_DIR, DATASETS_DIR, OUTPUTS_DIR, TMP_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

state = {
    "status": "idle",
    "step": "",
    "progress": 0,
    "message": "",
    "logs": [],
    "error": None,
    "result": None,
}
state_lock = threading.Lock()

# ---- Manual labeling session globals ----
# When run_pipeline reaches "waiting_manual_label", _MANUAL_LABEL_EVENT stays
# unsignaled (blocks pipeline thread) until the user clicks the UI "完成"
# button which calls /api/manual_label/complete → .set() wakes the thread.
_MANUAL_LABEL_EVENT = threading.Event()
_MANUAL_LABEL_CTX = {
    # Populated by run_pipeline before signaling the wait:
    "run_id": None,
    "frames_dir": None,   # str: Path to extracted raw JPGs (same dir train_server reads)
    "labels_dir": None,   # str: Path where YOLO .txt labels should be written
    "todo_filenames": [],  # list[str] basenames (frame_XXXX.jpg) of frames with NO .txt yet
    "done_filenames": set(),  # basenames saved OR skipped via API (so UI never re-prompts)
}

LLM_BASE_URL = "http://127.0.0.1:12345"
LLM_MODEL = "qwen3.5"


def llm_available():
    try:
        req = urllib.request.Request(f"{LLM_BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True
    except Exception:
        return False


def llm_chat(messages, temperature=0.3, max_tokens=3072):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw_body = resp.read().decode("utf-8")
        try:
            result = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"llm_chat: response from server is not valid JSON: {e}. "
                f"First 500 chars of body: {raw_body[:500]!r}"
            ) from e
    # Attach the raw response body to result so callers can dump it on empty-content failures.
    # We use a non-destructive dict key that won't conflict with the OpenAI schema.
    result["_raw_body"] = raw_body
    if "choices" not in result or not isinstance(result["choices"], list) or not result["choices"]:
        raise RuntimeError(
            f"llm_chat: response has no / empty 'choices' array. Full body first 800 chars: {raw_body[:800]!r}"
        )
    choice = result["choices"][0]
    msg = choice.get("message", choice)
    content = msg.get("content") if isinstance(msg, dict) else None
    finish = choice.get("finish_reason")
    # --- Diagnostics for empty / None content with non-standard finish_reason ---
    # llama.cpp multimodal bug manifests as: HTTP 200 + content="" (or null) +
    # finish_reason="abort"/"error"/"stop" even though 0 tokens were produced.
    # Previously we only checked finish_reason when content IS None; now we also
    # surface the raw response structure when content is empty/whitespace so
    # downstream failures (like "empty string") carry diagnostic breadcrumbs.
    content_is_empty = False
    if content is None:
        content_is_empty = True
    elif isinstance(content, str):
        if len(content.strip()) == 0:
            content_is_empty = True
    elif isinstance(content, list):
        # Multi-part content: no text parts at all → effectively empty.
        has_any_text = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                has_any_text = True
                break
        if not has_any_text:
            content_is_empty = True
    if content_is_empty:
        # Synthesize a content value that encodes the raw response so that
        # downstream code that records "raw_reply_preview" captures the cause.
        # We prepend a sentinel + finish_reason + key fields, then truncate to 4000 chars.
        snippet_parts = [
            f"[EMPTY_CONTENT_DEBUG] finish_reason={finish!r}",
        ]
        # --- Extra: detect reasoning-model quota exhaustion (qwen3.5 style)
        # When finish_reason='length' AND usage.completion_tokens >= max_tokens
        # AND content is empty, it means qwen3.5 spent ALL its quota writing
        # reasoning_content / "内心独白" before it even started emitting JSON.
        # Caller-side remediation: restart llama-server with `-c 8192` (or higher)
        # and ensure this llm_chat max_tokens default (3072) is well below
        # ctx-size minus the prompt tokens (~1500).
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            comp_toks = usage.get("completion_tokens")
            if (
                finish == "length"
                and isinstance(comp_toks, int)
                and comp_toks >= int(max_tokens)
            ):
                snippet_parts.insert(
                    1,
                    f"[QUOTA_EXHAUSTED] completion_tokens={comp_toks} hit max_tokens={max_tokens} → "
                    f"reasoning_content ate the whole budget before JSON could be written. "
                    f"REMEDY: (1) Restart llama-server with `-c 8192` at minimum (preferred); "
                    f"(2) prompt_tokens={usage.get('prompt_tokens')} → "
                    f"required ctx-size >= {int(comp_toks) + int(usage.get('prompt_tokens') or 0) + 512}.",
                )
        if isinstance(msg, dict):
            for k in ("tool_calls", "refusal", "reasoning_content"):
                if k in msg and msg[k] is not None:
                    snippet_parts.append(f"msg.{k}={str(msg[k])[:500]!r}")
        if isinstance(result, dict):
            for k in ("usage", "model", "id"):
                if k in result and result[k] is not None:
                    snippet_parts.append(f"resp.{k}={str(result[k])[:300]!r}")
        snippet_parts.append(f"raw_body[:2000]={raw_body[:2000]!r}")
        debug_snippet = " | ".join(snippet_parts)
        # Also raise if finish_reason clearly indicates an error
        if finish and finish not in ("stop", "eos", "length", None):
            # Refused / errored → raise with full context so caller gets a clear failure.
            raise RuntimeError(
                f"LLM abnormal finish: finish_reason={finish!r}. "
                f"Message keys: {list(msg.keys()) if isinstance(msg, dict) else type(msg)!r}. "
                f"Debug: {debug_snippet[:2000]}"
            )
        # For "stop"/"eos" / unknown / None finish_reason with empty content → don't
        # raise (preserves old behaviour of retrying via max_retries), but make the
        # content field NON-EMPTY with the debug snippet so raw_reply_preview and
        # failures sidecar files actually tell us *why* it was empty instead of "".
        content = debug_snippet
    # Normalize content (collapse list-of-content-parts, coerce None to empty-ish, etc.)
    # into either a plain str or None. Normalization is shared with the mock-friendly
    # helper llm_run so tests / callers get consistent behavior regardless of whether
    # llm_chat was patched by a mock that returns raw list / dict / None.
    return _normalize_llm_content(content, choice=choice, msg=msg)


def _normalize_llm_content(content, *, choice=None, msg=None):
    """Shared post-processing for raw LLM content returned from chat/completions.

    Accepts:
      None                        → None (preserves "LLM returned null" semantics)
      list of content-parts dicts → joined text portion, "" if no text parts
      any dict                    → try msg['content'] / msg['text'], else str()
      anything else               → str() coercion

    The normalization runs both for real llm_chat HTTP responses and for the value
    returned by any monkeypatched llm_chat in tests, so callers (health_probe,
    detect_boxes) can safely assume the output is either str or None.
    """
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    parts.append(c["text"])
                elif isinstance(c.get("text"), str):
                    parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    if content is None:
        return None
    if not isinstance(content, str):
        if isinstance(content, dict):
            alt = content.get("text") or content.get("content")
            if isinstance(alt, str):
                return alt
        return str(content)
    return content


_JSON_TRAILING_COMMA_RE = None
_JSON_MISSING_COMMA_RE = None
_JSON_LINE_COMMENT_RE = None
_JSON_BLOCK_COMMENT_RE = None


def _compile_json_regexes():
    """Lazily compile the small regex set used by _robust_json_parse."""
    global _JSON_TRAILING_COMMA_RE, _JSON_MISSING_COMMA_RE, _JSON_LINE_COMMENT_RE, _JSON_BLOCK_COMMENT_RE
    if _JSON_TRAILING_COMMA_RE is None:
        import re
        _JSON_TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')
        _JSON_MISSING_COMMA_RE = re.compile(r'([}\]])\s*([\[{])')
        _JSON_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
        _JSON_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', flags=re.DOTALL)


def _try_repair_json_string(candidate: str) -> str:
    """Apply lightweight, safe text-level repairs to an almost-valid JSON string.

    Order matters: comments first, then structural fixes, then trailing commas last.
    Each regex is safe even on fully-valid JSON (idempotent-ish). Never attempt to
    rewrite string contents (e.g. replace commas inside quotes); we only touch the
    structural punctuation that sits between JSON tokens.
    """
    _compile_json_regexes()
    s = candidate
    # 1) Strip // line comments and /* */ block comments (LLM sometimes
    #    appends " // note: I guessed the coords" style annotations).
    s = _JSON_BLOCK_COMMENT_RE.sub('', s)
    s = _JSON_LINE_COMMENT_RE.sub('', s)
    # 2) Missing comma between adjacent containers: "][", "}{", "]}", "}["
    #    e.g. `[[0.1,0.2,0.3,0.4] [0.5,0.6,0.7,0.8]]` → insert `,` between them.
    s = _JSON_MISSING_COMMA_RE.sub(r'\1,\2', s)
    # 3) Trailing comma before `}` or `]` — the #1 LLM JSON mistake.
    #    e.g. `[1, 2, 3,]` → `[1, 2, 3]`; `{"a": 1,}` → `{"a": 1}`
    s = _JSON_TRAILING_COMMA_RE.sub(r'\1', s)
    # 4) Second pass: after removing comments, the above two fixes may have
    #    introduced new fixable patterns (e.g. a comment sat between two arrays
    #    → comment stripped, now we have `][` → apply missing-comma again).
    s = _JSON_MISSING_COMMA_RE.sub(r'\1,\2', s)
    s = _JSON_TRAILING_COMMA_RE.sub(r'\1', s)
    return s


def _robust_json_parse(candidate: str):
    """json.loads with a repair fallback chain for LLM-typical malformations.

    Returns (parsed_obj, repair_note_or_None). On full failure raises a
    json.JSONDecodeError so callers retain a precise error message.
    """
    orig_err = None
    # Fast path: candidate is already valid JSON (the 95% case).
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        orig_err = e
    except Exception as e:
        # Defensive: any other failure types (e.g. UnicodeDecodeError on weird inputs)
        # wrap into JSONDecodeError-compatible form so downstream gets unified error path.
        orig_err = json.JSONDecodeError(
            f"Unexpected {type(e).__name__}: {e}", doc=str(candidate), pos=0,
        )
    # Slow path 1: text-level repairs then json.loads.
    repaired = _try_repair_json_string(candidate)
    if repaired != candidate:
        try:
            return json.loads(repaired), "text-repair"
        except json.JSONDecodeError as e:
            # Keep the original error for re-raise, unless the original had a
            # more user-visible character position.
            if orig_err is None:
                orig_err = e
    # Slow path 2: Python ast.literal_eval — it accepts trailing commas,
    # single quotes, and Python numeric literals. Only run it if the candidate
    # "looks like" a Python literal (starts with [ or {) to avoid surprising
    # string-coercion behaviour.
    stripped = candidate.strip()
    if stripped and stripped[0] in '[{':
        import ast
        try:
            val = ast.literal_eval(stripped)
            if isinstance(val, (dict, list)):
                return val, "ast.literal_eval"
        except (ValueError, SyntaxError, MemoryError, RecursionError) as _e:
            pass
        except Exception:
            pass
    # All fall-throughs exhausted. Re-raise the *original* error so the
    # caller-facing message still points at the actual user-visible syntax
    # issue (not something introduced by our repair attempts).
    if orig_err is not None:
        raise orig_err
    raise json.JSONDecodeError("Unable to parse candidate as JSON", doc=str(candidate), pos=0)


def _extract_and_normalize_json(text: str, *, expected_schema: str = "boxes"):
    """Shared robust JSON extractor for LLM replies. Handles:
       - Markdown code fences (```json ... ``` / ``` ... ```)
       - Extra prose / junk before or after the JSON payload
       - Bare array [[x1,y1,x2,y2], ...] replies (no outer {"boxes": ...} wrapper)
       - Nested object replies with a single known schema key.
       - LLM-typical JSON malformations: trailing commas, missing ][ commas,
         inline comments, single quotes (via ast.literal_eval fallback).

    Returns (obj_or_None, error_msg_or_None). The returned obj is always normalized
    so that the schema key (default "boxes") holds the list-of-boxes arrays.
    """
    if not text:
        return None, "empty text"
    t = text.strip()
    # Strip markdown code fences: ```json ... ``` / ``` ... ```
    if t.startswith("```"):
        # Strip opening fence (possibly with language tag like ```json / ```JSON)
        idx = t.find("\n")
        if idx >= 0:
            fence_line = t[:idx]
            # Accept if fence is just backticks or backticks+identifier
            if all(c == "`" or c.isalnum() or c == "_" or c == "-" for c in fence_line):
                t = t[idx + 1:]
        # Strip closing fence
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        # Some models output `json` tag without newline: ```json[[...]]```
        if t.lower().startswith("json"):
            t = t[4:].strip()
    if not t:
        return None, "empty after fence stripping"
    # Preferred path: find JSON object braces { ... }
    obj_start = t.find("{")
    obj_end = t.rfind("}")
    arr_start = t.find("[")
    arr_end = t.rfind("]")
    # Decide which payload to use first:
    #  - If object present and object starts before array (or no array): try object first
    #  - Else try array first, then fall back to object
    try_object_first = (
        obj_start >= 0
        and obj_end > obj_start
        and (arr_start < 0 or obj_start <= arr_start)
    )
    last_err = None
    repair_used = None
    if try_object_first:
        candidate = t[obj_start:obj_end + 1]
        try:
            obj, note = _robust_json_parse(candidate)
            repair_used = note or repair_used
            if isinstance(obj, dict):
                # Ensure the schema key exists; if dict has no schema key but has a single list value,
                # treat it as boxes (e.g. model returned {"result": [[...]]} instead of {"boxes": [[...]]}).
                if expected_schema not in obj:
                    for _, v in obj.items():
                        if (
                            isinstance(v, list)
                            and len(v) > 0
                            and isinstance(v[0], (list, tuple))
                            and len(v[0]) >= 4
                        ):
                            obj[expected_schema] = v
                            break
                if expected_schema not in obj:
                    obj[expected_schema] = []
                return obj, None
        except json.JSONDecodeError as e:
            last_err = f"object JSONDecodeError: {e}"
        # Object path failed; fall through to array path (if an array exists)
    if arr_start >= 0 and arr_end > arr_start:
        candidate = t[arr_start:arr_end + 1]
        try:
            arr, note = _robust_json_parse(candidate)
            repair_used = note or repair_used
            if isinstance(arr, list):
                # Either the array is a list of boxes (list of lists), or a single box (list of 4 numbers).
                # Normalize both into the dict schema form.
                if (
                    len(arr) > 0
                    and isinstance(arr[0], (list, tuple))
                ):
                    # List of boxes: [[0.1,0.2,0.3,0.4], ...]
                    return {expected_schema: arr}, None
                if (
                    len(arr) == 4
                    and all(isinstance(x, (int, float)) for x in arr)
                ):
                    # Single bare box: [0.1,0.2,0.3,0.4]
                    return {expected_schema: [arr]}, None
                return {expected_schema: []}, None
        except json.JSONDecodeError as e:
            last_err = (last_err + "; " if last_err else "") + f"array JSONDecodeError: {e}"
    # Fallback: try object path if we haven't already (array-started-first case)
    if not try_object_first and obj_start >= 0 and obj_end > obj_start:
        candidate = t[obj_start:obj_end + 1]
        try:
            obj, note = _robust_json_parse(candidate)
            repair_used = note or repair_used
            if isinstance(obj, dict):
                if expected_schema not in obj:
                    for _, v in obj.items():
                        if (
                            isinstance(v, list)
                            and len(v) > 0
                            and isinstance(v[0], (list, tuple))
                            and len(v[0]) >= 4
                        ):
                            obj[expected_schema] = v
                            break
                if expected_schema not in obj:
                    obj[expected_schema] = []
                return obj, None
        except json.JSONDecodeError as e:
            last_err = (last_err + "; " if last_err else "") + f"object JSONDecodeError: {e}"
    # Nothing parsed
    snippet = t[:120]
    err = last_err or f"no JSON object/array found in reply. snippet: {snippet!r}"
    if repair_used:
        err = f"{err} (repair fallback used: {repair_used})"
    return None, err


def llm_health_probe(sample_image_path, probe_prompt, expected_schema="boxes"):
    """Send a known tiny probe image + prompt to the local multimodal LLM and verify that
    it returns a parseable JSON payload matching the expected schema.

    Returns (ok: bool, detail: str) — ok=True only if the reply fully parses into the schema
    and is not empty / null / no-braces junk. When ok=False the detail string contains an
    actionable troubleshooting checklist for llama.cpp multimodal setups (the #1 failure is
    loading a text-only qwen3.5.gguf without --mmproj or the -VL variant gguf).
    """
    if sample_image_path is None:
        # Build a 2x2 solid-color JPEG in-memory so we don't need any on-disk fixture.
        import io
        import struct
        # Minimal 2x2 RGB JPEG payload (a single 8x8 MCU; content doesn't matter, we just
        # want the server-side vision pipeline to actually be exercised and reply non-empty).
        tiny = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00' + bytes([8]) * 64 +
            b'\xff\xc0\x00\x0b\x08\x00\x02\x00\x02\x01\x01\x11\x00'
            b'\xff\xc4\x00\x14\x00\x01' + bytes(18) +
            b'\xff\xc4\x00\x14\x10\x01' + bytes(18) +
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9'
        )
        tmp = io.BytesIO(tiny)
        b64 = base64.b64encode(tmp.getvalue()).decode("ascii")
    else:
        sample_image_path = Path(sample_image_path)
        with open(sample_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "Probe: you MUST reply with ONLY valid JSON following this exact schema, nothing else:\n"
                f'{{"{expected_schema}": [[x1, y1, x2, y2], ...]}}\n'
                "Rules for box elements:\n"
                "  - Every box MUST be a plain 4-element NUMBER array: [x1, y1, x2, y2].\n"
                "  - Never wrap a box as an OBJECT with nested keys like bbox / bbox_2d / points.\n"
                "  - Coordinates are NORMALIZED floats in [0.0, 1.0] from the image top-left corner.\n"
                f"User prompt: {probe_prompt}.\n"
                f'If no object is visible reply exactly {{"{expected_schema}": []}} with no commentary.'
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }]
    try:
        raw = llm_chat(messages, temperature=0.0, max_tokens=3072)
    except Exception as e:
        return False, (
            "🚨 本地 LLM 健康探测请求失败 (HTTP/chat 异常):\n"
            f"    {type(e).__name__}: {e}\n"
            "    请先确认 llama-server 能正常接收带图片的请求。"
        )
    # Ensure mock callers or weird server responses don't escape normalization.
    raw = _normalize_llm_content(raw)
    if raw is None:
        return False, (
            "🚨 本地 LLM 健康探测返回 null / None content。\n"
            "    常见原因: 当前模型是纯文本 gguf，不支持图片输入，但 capabilities 字段被服务端错误地标为了 multimodal。\n"
            "    另一个常见原因是请求的 max_tokens 太小（模型还没输出完就被截断为空），请重试。"
        )
    text = raw.strip()
    if not text:
        return False, _llm_vl_troubleshoot_text(
            "🚨 本地 LLM 健康探测返回空字符串（没有任何文字输出）。\n"
            "    很可能是 max_tokens 太小导致响应被截断；本代码已在健康探测中提高到 1024；\n"
            "    若仍为空，请检查 llama-server 侧是否有 token generation 报错 / 推理超时。",
            raw_len=0,
        )
    # Use the shared robust extractor: handles markdown fences + bare arrays + dict wrappers.
    obj, parse_err = _extract_and_normalize_json(text, expected_schema=expected_schema)
    if obj is None:
        snippet = text[:200]
        return False, _llm_vl_troubleshoot_text(
            f"🚨 本地 LLM 返回内容无法解析为有效 JSON (boxes schema)：{parse_err}\n"
            f"    实际回复 ({len(text)} chars):\n    {snippet!r}",
            raw_len=len(text),
            raw_snippet=snippet,
        )
    if expected_schema and not isinstance(obj, dict):
        return False, f"🚨 健康探测返回 JSON 顶层不是 object: {obj!r}"
    # Extra sanity check: if we got boxes, verify at least one coord looks like a normalized float
    # (if boxes is empty that's also valid — probe prompt may describe nothing in the 2x2 fixture).
    boxes = obj.get(expected_schema, [])
    if not isinstance(boxes, list):
        return False, f"🚨 健康探测返回 {expected_schema!r} 字段不是 list: {boxes!r}"
    return True, "健康探测 OK"


def _llm_vl_troubleshoot_text(summary, raw_len=None, raw_snippet=None):
    """Return a consolidated multi-line troubleshooting block for the #1 llama.cpp VL failure modes."""
    lines = [summary, "排查清单（按可能性从高到低）:"]
    lines.append(
        "  1) 模型是否真正加载了 VISION 版本？当前 /v1/models 显示 name=qwen3.5，通常意味着这是 TEXT-ONLY "
        "(纯文本) 8B/14B/32B gguf；你需要下载并加载文件名里明确带 -VL 的 gguf，例如 "
        "Qwen3.5-VL-8B-xxx.gguf (不是 Qwen3.5-8B-xxx.gguf)。"
    )
    lines.append(
        "  2) 是否传了 --mmproj？对于 llama.cpp 视觉模型，若视觉权重与 LLM 权重分两个 gguf，"
        "必须在启动 server 时加 --mmproj <qwen3.5-vl-mmproj-f16.gguf>；少了这个参数 LLM 会完全看不到图片，"
        "典型表现就是回复空字符串 / 只按文字 prompt 随便回答。"
    )
    lines.append(
        "  3) 检查 llama.cpp 启动日志：出现 'loaded mmproj' / 'multimodal adapter loaded' 之类字样才算成功启用视觉；"
        "如果 server 只打印了加载文本模型的日志，那就是参数 / 模型文件本身对不上。"
    )
    lines.append(
        "  4) llama.cpp 版本过旧？较新版本才在 /v1/chat/completions 的 OpenAI 兼容层里正确支持 "
        "'{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,...\"}}' 这种入参结构；"
        "老版本会静默丢弃图片内容。建议更新到近 1~2 个月内的 llama.cpp build。"
    )
    lines.append(
        "  5) 如使用 API-Proxy/其它中间件转发 12345，请确认它们不会剥离 content 数组里的 image_url 元素，"
        "只保留 text 部分（这也是常见的空回复来源）。"
    )
    if raw_len is not None:
        lines.append(f"  本次健康探测返回原始内容长度: {raw_len}")
    if raw_snippet:
        lines.append(f"  本次健康探测返回原始内容片段: {raw_snippet!r}")
    return "\n".join(lines)


def llm_detect_boxes(image_path, user_description, max_retries=2, *, temperature=None, extra_rules=None, extra_user_note=None):
    """Ask multimodal LLM directly for normalized 0..1 bounding boxes of the described target.

    Parameters (keyword-only for extensibility):
        temperature    - Override the default 0.0 sampling temperature. Higher values (0.3, 0.6) make
                         the model re-examine more carefully instead of anchoring on "no target".
        extra_rules    - Optional str appended to the "Rules (MANDATORY...)" section of the system
                         prompt. Used by caller-side retry passes to add per-pass emphasis rules.
                         The string is appended as a new bullet point, so callers should include the
                         leading "- ".
        extra_user_note - Optional str appended to the user prompt (after the description delimiter
                          block) as an extra emphasis note. Good for "this is retry pass 2, please
                          double-check even the tiny parts".

    Returns (boxes, raw_diagnostics_tuple) where boxes is a list of (x1,y1,x2,y2) tuples (normalized)
    and diagnostics tuple is (last_error: str|None, raw_reply_preview: str|None, n_attempts: int)
    so callers can summarize failures for debugging. On any failure boxes will be empty.
    """
    # Accept both str and Path; the debug log reads .name so we always normalize to Path.
    if not isinstance(image_path, Path):
        image_path = Path(image_path)

    # Pre-compute a bilingual prompt: keep user's original description + an English translation.
    # VL models often generalize better when the concept is expressed in English (their training
    # data is heavily English weighted), even when the user writes Chinese. We never modify the
    # user description itself; just append the English gloss as a disambiguation.
    english_gloss = _english_gloss_for(user_description)
    bilingual_desc = user_description
    if english_gloss and english_gloss.lower() != user_description.strip().lower():
        bilingual_desc = f"{user_description} (English reference for the VL model: {english_gloss})"

    system_prompt = (
        "You are a precise, single-pixel-level careful image annotator. "
        "You will be shown an image and a target object description in any language. "
        "Return ONLY a valid JSON object with this exact schema, and nothing else:\n"
        '{"boxes": [[x1, y1, x2, y2], ...]}\n\n'
        "Rules (MANDATORY, violations are fatal):\n"
        "- You MUST output at least one box if ANY portion of the described target is visible. "
        "This INCLUDES but is NOT LIMITED TO: partially occluded (e.g. covered by a hand, "
        "a finger, a sleeve, another object), blurry / out of focus, at the image border "
        "(cropped / cut off), or only a tiny sliver / arc / semicircle of a round cap is "
        "showing. Do NOT output {\"boxes\": []} just because the object is small, or only "
        "a small semicircle / partial circle is exposed — ALWAYS output its tight bounding "
        "box anyway. If a round lid / cap is blocked by a hand so only a 10% circular arc "
        "remains visible, you still MUST return the bounding box of the full underlying "
        "circle (or as much as you can infer). In Chinese: 哪怕只露出一小部分半圆 / 被手遮挡 "
        "/ 被其他物体挡住 / 只有一条弧形边露出 / 在画面边缘被裁掉了一部分，也必须框出来，"
        "绝对不能因为只看到一小块就返回空 boxes。\n"
        "- You MUST find every instance of the target. If there are 5 targets on screen "
        "(even if some are only partially visible), return 5 boxes; do not return only one.\n"
        "- Coordinates are NORMALIZED floats in [0.0, 1.0], measured from the TOP-LEFT image corner. "
        "x is width axis, y is height axis. x1=left, y1=top, x2=right, y2=bottom.\n"
        "- Each box must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.\n"
        "- Each box in 'boxes' MUST be a FLAT 4-element NUMBER ARRAY — do NOT wrap a box inside a "
        "JSON object with keys like 'bbox', 'bbox_2d', 'coords', 'points', 'xyxy', etc.\n"
        "- Box should TIGHTLY enclose the target, exclude unrelated background. If only a sliver "
        "of the target is visible, you may extrapolate to the full object outline if the shape "
        "is obvious (e.g. a clearly round cap with 90% covered still gets its full-circle box).\n"
        "- Only output {\"boxes\": []} when you are 100% confident NO matching object appears at all "
        "(e.g. image is blank or the description describes something clearly not present).\n"
        "- Do NOT include Markdown fences (```), explanations, text, keys other than 'boxes', or extra whitespace."
    )
    if extra_rules:
        # Inject caller-supplied emphasis rules right before the closing of the Rules list so the
        # LLM sees them in the same context window (and immediately after the core rules).
        if not system_prompt.endswith("\n"):
            system_prompt += "\n"
        system_prompt += (extra_rules if extra_rules.startswith("-") else "- " + extra_rules) + "\n"
    user_prompt_parts = [
        f"Target object description (any language, you must understand semantically):\n"
        f"---\n{bilingual_desc}\n---\n\n"
        "Return only {\"boxes\": [[x1,y1,x2,y2], ...]} with normalized coordinates for every matching instance. "
        'If nothing matches reply exactly {"boxes": []}.'
    ]
    if extra_user_note:
        user_prompt_parts.append("\n\n" + extra_user_note)
    user_prompt = "".join(user_prompt_parts)

    last_error = None
    last_reply_preview = None
    for attempt in range(max_retries + 1):
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt + "\n\n" + user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }]
            raw = llm_chat(messages, temperature=(temperature if temperature is not None else 0.0), max_tokens=3072)
            raw = _normalize_llm_content(raw)
            if raw is None:
                last_error = f"attempt {attempt+1}: LLM returned None"
                last_reply_preview = None
                continue
            # Keep up to 6000 chars (max_tokens=3072 → ~12KB typical) so failure diagnostics retain context
            last_reply_preview = raw[:6000]
            text = raw.strip()
            # Catch the [EMPTY_CONTENT_DEBUG] sentinel produced by llm_chat when the server
            # returned 0 tokens (llama.cpp multimodal KV cache overflow bug). Preserve the
            # user-visible "empty string" error classification so consecutive_empty counter
            # and had_llm_failure checks work as before, but attach the debug breadcrumbs
            # (finish_reason, usage, raw_body) that llm_chat stitched into the content.
            if text.startswith("[EMPTY_CONTENT_DEBUG]"):
                last_error = (
                    f"attempt {attempt+1}: LLM reply was empty string "
                    f"(server-side 0-token generation; llama.cpp VLP pipeline likely hit KV cache / ctx-size limit). "
                    f"Detail: {text[:2000]}"
                )
                continue
            # Defensive: guard against empty reply (e.g. LLM crashed, empty string response)
            if not text:
                last_error = f"attempt {attempt+1}: LLM reply was empty string"
                continue
            # Use shared robust extractor: markdown fences + bare-array fallback both handled here.
            obj, parse_err = _extract_and_normalize_json(text, expected_schema="boxes")
            if obj is None:
                last_error = f"attempt {attempt+1}: {parse_err}"
                # On the final attempt, attach the full reply snippet to the error message so
                # callers / failure sidecars capture the exact malformed content for inspection.
                if attempt == max_retries:
                    last_error = f"{last_error} | raw_content[:500]={text[:500]!r}"
                continue
            # obj is guaranteed to be a dict with a "boxes" key by the extractor.
            boxes = []
            for b in obj.get("boxes", []):
                coords = None
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    coords = b[:4]
                elif isinstance(b, dict):
                    # LLM sometimes wraps boxes in objects (despite system prompt). Try common keys:
                    #   bbox_2d, bbox, box, coords, coordinates, points, xyxy
                    for k in ("bbox_2d", "bbox", "box", "coords", "coordinates", "points", "xyxy"):
                        v = b.get(k)
                        if isinstance(v, (list, tuple)) and len(v) >= 4:
                            coords = v[:4]
                            break
                    if coords is None:
                        # Tolerate dict with plain numeric keys: {"0": x1, "1": y1, ...}
                        try:
                            coords = [float(b[str(i)]) for i in range(4)]
                        except Exception:
                            coords = None
                if coords is None:
                    continue
                try:
                    x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                except (ValueError, TypeError):
                    continue
                # Heuristic: detect pixel coordinates (e.g. the LLM returned 0..image size instead
                # of 0..1). If any value > 1.5 treat as pixel-scale / percent-scale; if all <=100
                # treat percent-scale and divide by 100; otherwise reject the single box.
                if any(v > 1.5 for v in (x1, y1, x2, y2)):
                    if all(v <= 100.0 for v in (x1, y1, x2, y2)):
                        x1, y1, x2, y2 = x1 / 100.0, y1 / 100.0, x2 / 100.0, y2 / 100.0
                    else:
                        last_error = (
                            f"attempt {attempt+1}: rejected box with out-of-range coords "
                            f"({x1:.3g},{y1:.3g},{x2:.3g},{y2:.3g}); expected 0..1 normalized"
                        )
                        continue
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    continue
                bw, bh = x2 - x1, y2 - y1
                if bw < 0.005 or bh < 0.005:
                    continue
                boxes.append((x1, y1, x2, y2))
            return boxes, (last_error, last_reply_preview, attempt + 1)
        except json.JSONDecodeError as e:
            last_error = f"attempt {attempt+1}: JSONDecodeError on reply: {e} (snippet: {text[:80]!r})"
            continue
        except Exception as e:
            last_error = f"attempt {attempt+1}: {type(e).__name__}: {e}"
            continue
    # All attempts exhausted
    print(f"[LLM] detect_boxes failed after {max_retries+1} tries on {image_path.name}: {last_error}")
    return [], (last_error, last_reply_preview, max_retries + 1)


def _english_gloss_for(user_description: str) -> str:
    """Very small Chinese->English glossary for common VL '瓶盖 / 北冰洋罐 / 汽车' style prompts.
    We intentionally only cover a handful of phrases because for arbitrary prompts this is out of
    scope; anything not covered returns empty string so callers skip the append.
    """
    text = user_description.strip()
    if not text:
        return ""
    gloss = text
    replacements = [
        ("红色的盖子", "red round cap / red lid (like a bottle cap)"),
        ("红色 圆形的盖子", "red round cap / red lid, circular bottle cap"),
        ("红色圆形盖子", "red round cap / red lid, circular bottle cap"),
        ("红色盖子", "red cap / red lid (bottle cap, can lid, etc.)"),
        ("盖子", "cap / lid (bottle cap, can lid)"),
        ("瓶盖", "bottle cap / can lid"),
        ("北冰洋", "Arctic Ocean brand soda can (cylindrical drink can, usually orange / red / yellow)"),
        ("北冰洋汽水", "Arctic Ocean orange soda can, cylindrical"),
        ("罐子", "metal can, soda can"),
        ("易拉罐", "aluminum soda / beverage can"),
        ("汽车", "car / automobile / vehicle"),
        ("红色汽车", "red car"),
        ("人", "person / human"),
        ("行人", "pedestrian / person"),
        ("猫", "cat"),
        ("狗", "dog"),
        ("杯子", "cup / mug"),
        ("瓶子", "bottle"),
        ("手机", "cell phone / smartphone"),
    ]
    for ch, en in replacements:
        if ch in gloss:
            # Don't double-substitute; just replace once
            gloss = gloss.replace(ch, en, 1)
    if gloss == text:
        return ""
    return gloss


def update_status(step, progress, message="", log=None):
    with state_lock:
        state["status"] = "running"
        state["step"] = step
        state["progress"] = progress
        state["message"] = message
        if log:
            state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {log}")
        state["error"] = None


def set_error(message):
    with state_lock:
        state["status"] = "error"
        state["error"] = message
        state["message"] = message
        state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")


def set_done(result=None):
    with state_lock:
        state["status"] = "done"
        state["progress"] = 100
        state["step"] = "complete"
        state["message"] = "Pipeline completed successfully!"
        if result:
            state["result"] = result
        state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline completed successfully!")


def get_status():
    with state_lock:
        return dict(state)


def reset_state():
    with state_lock:
        state["status"] = "idle"
        state["step"] = ""
        state["progress"] = 0
        state["message"] = ""
        state["logs"] = []
        state["error"] = None
        state["result"] = None
    # Also clean up any leftover manual-labeling globals so a new run starts fresh.
    _MANUAL_LABEL_EVENT.clear()
    _MANUAL_LABEL_CTX.update({
        "run_id": None,
        "frames_dir": None,
        "labels_dir": None,
        "todo_filenames": [],
        "done_filenames": set(),
    })


def extract_frames(video_path, output_dir, fps=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(round(video_fps / fps))) if video_fps > 0 else 1
    extracted = 0
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            name = f"frame_{saved_idx + 1:04d}"
            cv2.imwrite(str(output_dir / f"{name}.jpg"), frame)
            extracted += 1
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return extracted, width, height


def auto_label_frames(frames_dir, labels_dir, class_name):
    """LLM-direct labeling: each frame is sent to the multimodal LLM together with the user's
    description (any language). The LLM returns normalized bounding boxes directly, which we
    convert into YOLO-format (cx, cy, bw, bh) label files. No YOLO-World, no probes, no retries."""

    image_files = sorted(frames_dir.glob("*.jpg"))
    total = len(image_files)

    if not llm_available():
        raise RuntimeError(
            "LLM 服务 http://127.0.0.1:12345 不可用。"
            "当前版本默认用大模型逐帧标注，请先启动本地 LLM 再试。"
        )

    update_status(
        "init_model",
        5,
        f"使用大模型直接标注 (共 {total} 帧)",
        f"LLM 标注模式: 逐帧发送图片 + 用户描述 '{class_name}' -> LLM 返回 JSON boxes 坐标",
    )

    # ------------------------------------------------------------------
    # Health probe (fail-fast): send the first real frame to the LLM before
    # starting the N-frame loop. If the server is running a TEXT-ONLY model
    # (e.g. user loaded qwen3.5.gguf without the -VL variant / without
    # --mmproj) we'll get an empty / junk reply and can abort immediately
    # with actionable diagnostics instead of burning N retries × N frames.
    # ------------------------------------------------------------------
    sample = image_files[0] if image_files else None
    probe_desc = class_name or "any visible object"
    probe_ok, probe_detail = llm_health_probe(sample, probe_desc, expected_schema="boxes")
    if not probe_ok:
        print(probe_detail)
        update_status(
            "failed", 0, "LLM 健康探测失败，已中止。",
            probe_detail,
        )
        raise RuntimeError(
            "LLM 健康探测失败（本地大模型看起来没有真正加载视觉能力）。\n"
            + probe_detail
        )
    update_status(
        "labeling",
        7,
        f"LLM 健康探测通过，开始逐帧标注 ({total} frames)",
    )

    labeled = 0
    total_boxes = 0
    parse_errors = 0
    empty_frames = 0
    llm_failures = 0          # LLM communication / parse / empty-reply failures (not "no object visible" cases)
    # Post-process retry stats: the extra 2 passes for truly-empty frames to squeeze more recall.
    r1_frames = 0             # pass 1 (default) first-try: how many frames got boxes on pass 1
    r2_retry_count = 0        # how many frames went into pass 2
    r2_salvaged = 0           # how many of them got boxes via pass 2
    r3_retry_count = 0        # how many still-empty frames went into pass 3
    r3_salvaged = 0           # how many of them got boxes via pass 3
    r13_all_empty = 0         # how many stayed empty after all 3 passes
    diagnostics = []          # (frame_name, last_error, raw_reply_preview) for failures
    consecutive_empty = 0
    CONSECUTIVE_EMPTY_ABORT = 10  # after this many consecutive fully-empty LLM replies, abort pipeline
    consecutive_broken = 0
    CONSECUTIVE_BROKEN_ABORT = 5  # after this many consecutive parse / empty-string / None reply, abort
    MAX_EMPTY_RETRY_PASSES = 2    # user-requested: up to 2 additional runs for "no result" frames (pass 2 & 3)

    # Output dir for copies of images that failed LLM reply + their diagnostic txt. User can inspect
    # them visually after a run, or use them to re-test llm_detect_boxes manually.
    failures_dir = labels_dir.parent / "failures"
    if failures_dir.exists():
        for p in list(failures_dir.iterdir()):
            p.unlink()
    failures_dir.mkdir(exist_ok=True)

    t_total_start = time.time()

    for idx, img_path in enumerate(image_files):
        t0 = time.time()
        boxes, diag = llm_detect_boxes(str(img_path), class_name, max_retries=2)
        (last_error, raw_preview, _n_attempts) = diag
        final_pass_used = 1  # pass 1 is the default run
        dt = time.time() - t0

        had_llm_failure = bool(last_error and "empty string" in last_error) or bool(
            last_error and ("returned None" in last_error or "JSONDecodeError" in last_error or "no JSON object" in last_error or "found in reply" in last_error)
        )
        if had_llm_failure:
            llm_failures += 1
            consecutive_broken += 1
            consecutive_empty += 1 if (last_error and "empty string" in last_error) else 0
            diagnostics.append((img_path.name, last_error or "", raw_preview or ""))
            # Save a diagnostic copy: JPG + sidecar .txt + full raw reply .raw.txt
            try:
                shutil.copy2(img_path, failures_dir / img_path.name)
                sidecar = failures_dir / f"{img_path.stem}.txt"
                lines = [
                    f"user_description: {class_name}",
                    f"final_pass_used: {final_pass_used}",
                    f"last_error: {last_error or ''}",
                    f"raw_reply_first_4000_chars:",
                    raw_preview or '',
                ]
                sidecar.write_text("\n".join(lines), encoding="utf-8")
                # Also dump the full raw reply to a standalone file for copy-paste / inspection.
                if raw_preview:
                    raw_file = failures_dir / f"{img_path.stem}.raw_reply.txt"
                    raw_file.write_text(raw_preview, encoding="utf-8")
            except Exception:
                pass
            # Abort thresholds — if we're clearly not getting valid replies anymore, stop early
            # instead of burning minutes while the OOM'd / frozen server returns empty strings.
            if consecutive_broken >= CONSECUTIVE_BROKEN_ABORT:
                summary_err = (
                    f"LLM 连续 {CONSECUTIVE_BROKEN_ABORT} 帧都失败（最近一次: {last_error}）。"
                    "这通常意味着 llama-server 显存占用过高 / OOM 导致推理失败或超时。\n"
                    "修复建议: 1) 重启 llama-server；2) 降低 -b / -ngl / --mlock 等减少显存占用；"
                    "3) 确认 -c 上下文长度足够放图像占位 token；4) 若视频太长，降低抽帧 fps。"
                )
                update_status("failed", 0, "LLM 连续失败，已中止标注。", summary_err)
                raise RuntimeError(summary_err)
            if consecutive_empty >= CONSECUTIVE_EMPTY_ABORT:
                summary_err = (
                    f"LLM 连续 {CONSECUTIVE_EMPTY_ABORT} 帧返回空字符串（reply was empty string）。"
                    "典型原因: llama-server 视觉编码管线崩了（mmproj 加载错误 / OOM）。\n"
                    "修复建议: 重启 llama-server 并重试；如仍然失败，降低抽帧 fps。"
                )
                update_status("failed", 0, "LLM 连续空回复，已中止标注。", summary_err)
                raise RuntimeError(summary_err)
        else:
            consecutive_broken = 0
            consecutive_empty = 0
            # ==========================================================
            # USER REQUESTED RETRY LOGIC: up to 2 extra passes for
            # frames that returned boxes=[] (truly empty) on pass 1.
            # ==========================================================
            if not boxes:
                # Retry pass 2: temperature=0.3, append aggressive rule asking the LLM to
                # re-examine extremely small slivers / partial circles / hand-shielded regions.
                r2_retry_count += 1
                t_p2 = time.time()
                boxes_r2, diag_r2 = llm_detect_boxes(
                    str(img_path), class_name, max_retries=1,
                    temperature=0.3,
                    extra_rules=(
                        "- EXTREMELY IMPORTANT RE-EXAMINATION RULE (retry pass 2 / 3): "
                        "The previous run for this same image returned boxes=[] which was likely "
                        "wrong. Re-examine every 1x1 pixel patch with extreme care. Look for the "
                        "target object EVEN IF IT IS: (a) only a tiny sliver of red semicircle/arc "
                        "visible on the image border, (b) covered by a hand / finger so only the "
                        "edge curve is showing, (c) smaller than 5% of the image area, "
                        "(d) so blurry that you can barely make out the circular outline. "
                        "漏检（false negative）比误检严重得多。In Chinese: 请极其仔细地重新检查整张图片，"
                        "特别是画面边缘和被手/手指/其他物体遮挡的局部区域；只要有任何一丝、一条"
                        "弧形边、1/4个圆形、一个小红点看起来像目标，都必须框出来，绝不允许因为只有"
                        "极小一部分就返回空 boxes。漏检比误检的后果要严重得多。\n"
                    ),
                    extra_user_note=(
                        "This is RETRY PASS 2/3 for this image. My previous answer to the same "
                        "question was boxes=[] and it was almost certainly a false negative. "
                        "Overrule your previous confidence: if there is even a 10% chance the "
                        "target appears anywhere (partially visible, tiny, obscured), output the "
                        "box instead of returning empty. / 这是对这张图的重试第 2/3 轮，请推翻之前"
                        "的判断，哪怕只有一成把握有目标也必须框出来。"
                    ),
                )
                dt += (time.time() - t_p2)
                (r2_last_err, r2_raw_prev, _r2_att) = diag_r2
                if boxes_r2:
                    boxes = boxes_r2
                    last_error = r2_last_err  # may be None or leftover warning; r2 actually succeeded
                    raw_preview = r2_raw_prev
                    final_pass_used = 2
                    r2_salvaged += 1
                    # Override had_llm_failure for r2-only if r2 had a failure (shouldn't happen
                    # when boxes_r2 is non-empty, but guard anyway).
                    had_llm_failure = bool(r2_last_err and ("empty string" in r2_last_err or
                        "returned None" in r2_last_err or "JSONDecodeError" in r2_last_err or
                        "no JSON object braces" in r2_last_err))
                else:
                    # Retry pass 3 / 3: temperature=0.6, more aggressive.
                    r3_retry_count += 1
                    t_p3 = time.time()
                    boxes_r3, diag_r3 = llm_detect_boxes(
                        str(img_path), class_name, max_retries=1,
                        temperature=0.6,
                        extra_rules=(
                            "- CRITICAL FINAL RE-EXAMINATION RULE (retry pass 3 / 3): "
                            "This is your LAST CHANCE on this image. There is VERY STRONG "
                            "prior belief that the target is present in this image even if "
                            "you can't see it clearly. You MUST output at least one box "
                            "UNLESS the image truly contains nothing that could possibly be "
                            "the target. In Chinese: 这是最后一次重试机会。此帧图片在数据集里"
                            "极大概率确实包含你要找的目标，请再放大想象细看画面边缘、被手遮挡"
                            "的局部、模糊的弧形边缘、只有一小条红色圆弧的地方；只要有任何可能"
                            "是目标就必须框出来，不到万不得已不要返回空。"
                        ),
                        extra_user_note=(
                            "FINAL RETRY PASS 3/3. Overrule your prior two empty-box decisions. "
                            "It is MUCH better to return a slightly-off box than to miss the "
                            "target entirely. / 这是最后一次重试，推翻前两次的空判断，漏检远"
                            "比误检糟糕，请宁可多框一个可疑区域也不要返回空。"
                        ),
                    )
                    dt += (time.time() - t_p3)
                    (r3_last_err, r3_raw_prev, _r3_att) = diag_r3
                    if boxes_r3:
                        boxes = boxes_r3
                        last_error = r3_last_err
                        raw_preview = r3_raw_prev
                        final_pass_used = 3
                        r3_salvaged += 1
                        had_llm_failure = bool(r3_last_err and ("empty string" in r3_last_err or
                            "returned None" in r3_last_err or "JSONDecodeError" in r3_last_err or
                            "no JSON object braces" in r3_last_err))
            if not boxes and not had_llm_failure:
                empty_frames += 1  # truly no object (reply boxes: []), not a failure (after all 3 passes)
                r13_all_empty += 1
            elif boxes and final_pass_used == 1:
                r1_frames += 1
            if not boxes:
                # Save per-frame diagnostic for empty frames (whether truly-empty or LLM failed)
                # so user can see which passes were tried and what the last reply looked like.
                try:
                    shutil.copy2(img_path, failures_dir / img_path.name)
                    sidecar = failures_dir / f"{img_path.stem}.txt"
                    lines = [
                        f"user_description: {class_name}",
                        f"final_pass_used: {final_pass_used}",
                    ]
                    if r2_retry_count or final_pass_used >= 2:
                        lines += [
                            f"r2_was_tried: {'yes' if r2_retry_count else 'no'}",
                            f"r3_was_tried: {'yes' if (r3_retry_count and final_pass_used in (1,3)) or final_pass_used == 3 else 'no'}",
                        ]
                    lines += [
                        f"last_error: {last_error or ''}",
                        f"raw_reply_preview: {raw_preview or ''}",
                    ]
                    sidecar.write_text("\n".join(lines), encoding="utf-8")
                except Exception:
                    pass

        label_path = labels_dir / f"{img_path.stem}.txt"
        if boxes:
            with open(label_path, "w") as f:
                for x1, y1, x2, y2 in boxes:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = x2 - x1
                    bh = y2 - y1
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            labeled += 1
            total_boxes += len(boxes)
            extra_parts = []
            if final_pass_used >= 2:
                extra_parts.append(f"pass{final_pass_used} salvaged")
            if last_error:
                extra_parts.append(f"last_err={last_error}")
            extra = f" ({'; '.join(extra_parts)})" if extra_parts else ""
            log_msg = f"Labeled: {img_path.name} ({len(boxes)} box, {dt:.1f}s){extra}"
        else:
            # write empty file (consistent with YOLO negatives)
            label_path.write_text("")
            # empty_frames incremented once in the pass1/2/3 block above to avoid double-counting.
            if had_llm_failure:
                log_msg = (
                    f"Failed: {img_path.name} — {last_error} ({dt:.1f}s)"
                )
                parse_errors += 1
            else:
                tries = f" (3-passes empty, r2/r3 tried)" if final_pass_used >= 2 else ""
                log_msg = f"Skipped: {img_path.name} — no matching object ({dt:.1f}s){tries}"

        progress = int((idx + 1) / total * 100)
        update_status(
            "labeling",
            progress,
            f"LLM 标注 {idx + 1}/{total}",
            log_msg,
        )

    # Emit a consolidated per-run diagnostic report so the user can eyeball which frames went bad
    # without scrolling through 64 lines of log.
    report_path = labels_dir.parent / "labeling_report.txt"
    lines = [
        f"user_description: {class_name}",
        f"total_frames: {total}",
        f"labeled_frames: {labeled}",
        f"  - pass1 (default, temperature=0.0) first-hit labeled: {r1_frames}",
        f"  - pass2 (retry, temperature=0.3) salvaged (was empty on pass1): {r2_salvaged} / {r2_retry_count}",
        f"  - pass3 (retry, temperature=0.6) salvaged (was still empty after pass2): {r3_salvaged} / {r3_retry_count}",
        f"  - all 3 passes empty (still no boxes): {r13_all_empty}",
        f"totally_empty_frames (no-object reply + failure frames): {empty_frames + llm_failures}",
        f"truly_empty_frames (boxes=[] after all 3 passes, no LLM failure): {empty_frames}",
        f"llm_failure_frames (empty-string / None / parse error): {llm_failures}",
        f"total_boxes: {total_boxes}",
        f"total_seconds: {time.time() - t_total_start:.1f}",
        "",
    ]
    if diagnostics:
        lines.append(f"failure details (n={len(diagnostics)}):")
        for name, err, preview in diagnostics:
            lines.append(f"  - {name}: {err}")
            if preview:
                lines.append(f"      reply_preview: {preview[:400]}")
    else:
        lines.append("No LLM failures detected this run.")
    try:
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[auto_label_frames] Report written to: {report_path}")
    except Exception:
        pass

    total_dt = time.time() - t_total_start
    update_status(
        "labeling",
        100,
        f"LLM 标注完成: {labeled}/{total} 帧有标注 (共 {total_boxes} 个框, {total_dt:.0f}s)",
        (
            f"有标注 {labeled} (首轮命中 {r1_frames}, 第2轮救回 {r2_salvaged}/{r2_retry_count}, "
            f"第3轮救回 {r3_salvaged}/{r3_retry_count}), 真正空帧(3轮都没物体) {empty_frames}, "
            f"LLM故障帧 {llm_failures}, 总框数 {total_boxes}, 总耗时 {total_dt:.1f}s, "
            f"平均 {total_dt/max(1,total):.1f}s/帧。详细报告: {report_path.name}；"
            f"失败图片和侧车 txt 在: {failures_dir.name}/"
        ),
    )

    return labeled


def split_dataset(images_dir, labels_dir, train_ratio=0.9):
    image_files = sorted(images_dir.glob("*.jpg"))
    total = len(image_files)
    split_idx = int(total * train_ratio)

    train_images = images_dir / "train"
    val_images = images_dir / "val"
    train_labels = labels_dir / "train"
    val_labels = labels_dir / "val"

    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_files):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if i < split_idx:
            dest_img = train_images / img_path.name
            dest_lbl = train_labels / lbl_path.name
        else:
            dest_img = val_images / img_path.name
            dest_lbl = val_labels / lbl_path.name

        shutil.copy2(str(img_path), str(dest_img))
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(dest_lbl))


def generate_yaml(dataset_dir, class_name, yaml_path):
    config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    # Explicit UTF-8 so read_class_names (which reads UTF-8 below) can round-trip
    # Chinese class names. Without this, Windows Python defaults to the active
    # code page (e.g. GBK on zh-CN), and the UTF-8 reader drops/garbles chars.
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _read_text_any_encoding(path):
    """Read a text file, trying UTF-8 first, then GBK/GB18030 on decode error.

    Legacy dataset.yaml files (written before generate_yaml had explicit
    UTF-8) defaulted to the Windows code page (GBK on zh-CN systems), so a
    strict UTF-8 reader would silently drop every Chinese character via
    errors="ignore" and return an empty class list. Trying GBK as fallback
    lets us round-trip both encodings cleanly.
    """
    data = Path(path).read_bytes()
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_class_names(yaml_path):
    """Return class names as an ordered list from a YOLO dataset.yaml (names: {id: name})."""
    yaml_path = Path(yaml_path)
    try:
        text = _read_text_any_encoding(yaml_path)
        cfg = yaml.safe_load(text) or {}
        names = cfg.get("names")
        if isinstance(names, list):
            return [str(n).strip() for n in names if str(n).strip()]
        if isinstance(names, dict):
            items = sorted((int(k), str(v).strip()) for k, v in names.items())
            return [v for _, v in items if v]
    except Exception:
        pass
    return []


def post_train_export(pt_path, yaml_path, output_dir, android_assets_dir=None, imgsz=640):
    """Auto-export newly-trained .pt -> .onnx -> .tflite alongside labels.txt.

    Runs after every training run. Failures are non-fatal (they do NOT mark the
    overall training run as failed) — detailed logs are written so the user can
    retry manually via `python export_to_tflite.py` if anything goes wrong.

    Produces in output_dir/:
      - detect.tflite
      - labels.txt
    If android_assets_dir is given the files are also copied there so a
    subsequent gradlew build immediately packages the new weights.
    """
    pt_path = Path(pt_path)
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        log_lines.append(str(msg))
        print("[export]", msg)

    log(f"Post-train export starting. pt={pt_path.name} out={output_dir}")
    overall_ok = True

    # ---- labels.txt ----
    try:
        classes = read_class_names(yaml_path)
        if not classes:
            # Fallback: parse names block manually with regex so even weird
            # yaml loader quirks don't kill label generation.
            import re
            # Use same tolerant reader: legacy yaml files may be GBK-encoded.
            text = _read_text_any_encoding(yaml_path)
            found = {}
            in_names = False
            for line in text.splitlines():
                s = line.rstrip()
                if s.startswith("names:"):
                    in_names = True
                    continue
                if in_names and s and not s[:1].isspace() and not s.startswith("#"):
                    break
                m = re.match(r"\s*(\d+)\s*:\s*(.+)", s)
                if m:
                    found[int(m.group(1))] = m.group(2).strip()
            classes = [v for _, v in sorted(found.items())]
        if not classes:
            classes = ["object"]
            log("WARN: could not resolve class names from yaml; falling back to ['object']")
        labels_out = output_dir / "labels.txt"
        labels_out.write_text("\n".join(classes) + "\n", encoding="utf-8")
        log(f"labels.txt -> {labels_out} ({len(classes)} classes: {classes})")
    except Exception as e:
        overall_ok = False
        log(f"ERROR writing labels.txt: {type(e).__name__}: {e}")
        labels_out = None

    # ---- .pt -> ONNX ----
    onnx_path = None
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        onnx_out = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
        onnx_path = Path(onnx_out)
        log(f"ONNX export OK -> {onnx_path.name} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")
    except Exception as e:
        overall_ok = False
        log(f"ERROR onnx export: {type(e).__name__}: {e}")

    # ---- ONNX -> TFLite ----
    tflite_final = None
    if onnx_path and onnx_path.exists():
        # See export_to_tflite.py SYS_PY_WITH_TF_AND_ONNX2TF comment for the
        # full rationale.  Short version:
        #   - Use SYSTEM python (has tensorflow+onnx2tf), not project .venv
        #   - onnx2tf 1.22 default flags (no -osd, no -k) preserves YOLO head
        #     perfectly (float32 model matches raw PT output bit-for-bit).  The
        #     keep-op flags from an earlier attempt were only masking a
        #     DIFFERENT failure mode (onnx2tf crash from TRAE sandbox writing
        #     __pycache__ under site-packages).
        #   - Five runtime-protection layers: PYTHONDONTWRITEBYTECODE=1, wipe
        #     onnx2tf __pycache__, regenerate calibration npy, allow_pickle
        #     patch, and CPU-only TF env.  These were validated on the orange
        #     can reference model (20260805_095916) where the float32 tflite
        #     reproduces frame_0001 conf=0.79 vs PT's conf=0.87 exactly.
        import subprocess
        SYS_PY_WITH_TF_AND_ONNX2TF = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"

        def _heal_allow_pickle(sys_py: str) -> None:
            """Equivalent of export_to_tflite._ensure_onnx2tf_allow_pickle."""
            probe = subprocess.run(
                [sys_py, "-c",
                 "import onnx2tf.utils.common_functions as m, os; "
                 "print(os.path.abspath(m.__file__))"],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                log(f"  [self-heal L4] cannot locate onnx2tf; rc={probe.returncode}")
                return
            target = Path(probe.stdout.strip())
            if not target.exists():
                return
            src = target.read_text(encoding="utf-8")
            OLD = "test_image_data: np.ndarray = np.load(f)\n"
            NEW = ("# NOTE: cache files fetched on older numpy versions contained pickled dtype metadata; "
                   "numpy >=1.26 flipped allow_pickle=False by default. Explicit True restores behavior.\n"
                   "        test_image_data: np.ndarray = np.load(f, allow_pickle=True)\n")
            if OLD not in src and "allow_pickle=True" not in src:
                log(f"  [self-heal L4] allow_pickle patch point not found in {target.name}; skip.")
                return
            if OLD in src:
                patched = src.replace(OLD, NEW, 1)
                target.write_text(patched, encoding="utf-8")
                log(f"  [self-heal L4] onnx2tf allow_pickle=True applied ({target.name}).")
            else:
                log(f"  [self-heal L4] allow_pickle already patched ({target.name}); skip.")

        def _prepare_env_and_fs(cwd_for_subprocess: Path) -> dict:
            """Equivalent of export_to_tflite._prepare_onnx2tf_runtime (5 layers)."""
            import os as _os
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"                # L1
            env["CUDA_VISIBLE_DEVICES"] = "-1"                   # L5
            env["TF_ENABLE_ONEDNN_OPTS"] = "0"                   # L5
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            env["YOLO_AUTOINSTALL"] = "False"

            # L2: wipe onnx2tf utils __pycache__
            _probe = subprocess.run(
                [SYS_PY_WITH_TF_AND_ONNX2TF, "-c",
                 "import onnx2tf.utils.common_functions as m, os, shutil; "
                 "p=os.path.join(os.path.dirname(os.path.abspath(m.__file__)),'__pycache__'); "
                 "shutil.rmtree(p, ignore_errors=True); print('ok')"],
                capture_output=True, text=True,
            )
            if _probe.returncode == 0 and _probe.stdout.strip() == "ok":
                log("  [self-heal L2] onnx2tf utils __pycache__ wiped.")
            else:
                log(f"  [self-heal L2] __pycache__ wipe note: rc={_probe.returncode}")

            # L3: write calibration npy into `cwd_for_subprocess` (onnx2tf
            # checks getcwd() for it first).  Writing via the SAME interpreter
            # guarantees the numpy save/load format round-trips cleanly (the
            # v1.20.4 file onnx2tf would otherwise download was pickled by
            # numpy <1.26 and newer numpy raises UnpicklingError even with
            # allow_pickle=True).
            cwd_for_subprocess.mkdir(parents=True, exist_ok=True)
            calib_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
            calib_path = cwd_for_subprocess / calib_name
            _w = subprocess.run(
                [SYS_PY_WITH_TF_AND_ONNX2TF, "-c",
                 "import sys, numpy as np; "
                 "p=sys.argv[1]; "
                 "np.save(p, (np.random.rand(20,128,128,3).astype(np.float32)*255.0)); "
                 "a=np.load(p, allow_pickle=False); "
                 "b=np.load(p, allow_pickle=True); "
                 "print(a.shape, a.dtype, a.min(), a.max(), 'BOTH_OK')",
                 str(calib_path)],
                capture_output=True, text=True,
            )
            if _w.returncode == 0 and "BOTH_OK" in _w.stdout:
                log(f"  [self-heal L3] wrote valid calibration cache -> {calib_name} "
                    f"({calib_path.stat().st_size/1024:.0f} KB)")
            else:
                log(f"  [self-heal L3] warn cache writer rc={_w.returncode}")
            return env

        # L4 + L1/L2/L3/L5
        _heal_allow_pickle(SYS_PY_WITH_TF_AND_ONNX2TF)
        # ASCII-only temporary output path for onnx2tf.  tensorflow 2.15's
        # TFLite Interpreter / onnx2tf file writer has unicode-path bugs on
        # Windows; the final copy-to-destination handles unicode names fine.
        tflite_ascii_tmp_dir = output_dir / "_tflite_ascii_tmp"
        if tflite_ascii_tmp_dir.exists():
            try:
                shutil.rmtree(tflite_ascii_tmp_dir, ignore_errors=True)
            except Exception:
                pass
        # onnx2tf checks getcwd() for calibration_image_sample_data_*.npy so
        # run the subprocess with cwd=OUTPUTS_DIR (parent of ascii tmp dir).
        subprocess_cwd = output_dir
        subprocess_env = _prepare_env_and_fs(subprocess_cwd)

        # NOTE: NO `-osd` and NO `-k <keep_ops>`.  onnx2tf 1.22's default
        # op-fusion rules produce a float32.tflite that matches raw PT output
        # bit-for-bit on the reference orange-can model.  The old keep-op
        # flags + -osd were noise that hid the *actual* crash root cause
        # (__pycache__ write + pickled-calendar-npy incompatibility).
        onnx2tf_cmd = [
            SYS_PY_WITH_TF_AND_ONNX2TF, "-m", "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(tflite_ascii_tmp_dir),
        ]
        log(f"Running onnx2tf: {' '.join(map(str, onnx2tf_cmd))}")
        log(f"  env[PYTHONDONTWRITEBYTECODE]={subprocess_env.get('PYTHONDONTWRITEBYTECODE')} "
            f"CUDA_VISIBLE_DEVICES={subprocess_env.get('CUDA_VISIBLE_DEVICES')} "
            f"cwd={subprocess_cwd}")
        try:
            res = subprocess.run(
                onnx2tf_cmd,
                cwd=str(subprocess_cwd),
                env=subprocess_env,
                capture_output=True, text=True, timeout=60 * 60,
            )
            if res.returncode != 0:
                tail = (res.stdout or "")[-1800:] + "\n---STDERR---\n" + (res.stderr or "")[-2500:]
                raise RuntimeError(f"onnx2tf exit={res.returncode}\n{tail}")
            candidates = list(tflite_ascii_tmp_dir.rglob("*.tflite"))
            if not candidates:
                raise RuntimeError(f"onnx2tf produced no .tflite in {tflite_ascii_tmp_dir}")
            # Prefer _float32 > _float16 > anything else
            ordered = sorted(
                candidates,
                key=lambda p: (0 if "_float32" in p.name
                               else 1 if "_float16" in p.name else 2,
                               p.stat().st_size),
            )
            tflite_src = ordered[0]
            tflite_final = output_dir / "detect.tflite"
            shutil.copy2(tflite_src, tflite_final)
            log(f"TFLite fp32 OK -> {tflite_final.name} ({tflite_final.stat().st_size/1024/1024:.1f} MB) "
                f"[source: {tflite_src.name}]")
            # Also ship the fp16 variant as optional backup asset next to main one
            for p in ordered:
                if "_float16" in p.name:
                    fp16_dst = output_dir / "detect_float16.tflite"
                    shutil.copy2(p, fp16_dst)
                    log(f"TFLite fp16 OK -> {fp16_dst.name} ({fp16_dst.stat().st_size/1024/1024:.1f} MB)")
                    break
            try:
                shutil.rmtree(tflite_ascii_tmp_dir, ignore_errors=True)
            except Exception:
                pass
        except Exception as e:
            overall_ok = False
            log(f"ERROR tflite conversion: {type(e).__name__}: {e}")

    # ---- Copy to Android assets ----
    copied_to_android = False
    if android_assets_dir and (tflite_final or labels_out):
        try:
            android_assets_dir = Path(android_assets_dir)
            android_assets_dir.mkdir(parents=True, exist_ok=True)
            if tflite_final and tflite_final.exists():
                shutil.copy2(tflite_final, android_assets_dir / "detect.tflite")
                copied_to_android = True
                # Also ship the fp16 variant alongside so the Android app can
                # fall back to it if fp32 is too large for some devices.
                fp16_src = output_dir / "detect_float16.tflite"
                if fp16_src.exists():
                    shutil.copy2(fp16_src, android_assets_dir / "detect_float16.tflite")
            if labels_out and labels_out.exists():
                shutil.copy2(labels_out, android_assets_dir / "labels.txt")
                copied_to_android = True
            if copied_to_android:
                log(f"Copied new model+labels into Android assets: {android_assets_dir}")
        except Exception as e:
            log(f"WARN copying to android assets failed: {type(e).__name__}: {e}")

    # Write consolidated sidecar report so user can eyeball what happened
    report = output_dir / f"export_report_{pt_path.stem}.txt"
    try:
        report.write_text(
            "\n".join(log_lines) +
            f"\n\noverall: {'SUCCESS' if overall_ok else 'FAILED (see errors above)'}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    return overall_ok, tflite_final, labels_out, copied_to_android


# ---------------------------------------------------------------------------
# Post-train smoke test: run the fresh PT + TFLite model on every image in
# TEST_DIR (ml/test/*.jpg).  Produces side-by-side annotated PNGs and a
# short report so the user can immediately eyeball whether the model can
# actually detect the object on real-world photos (not just training frames).
# ---------------------------------------------------------------------------
def _draw_boxes_cv2(img_bgr, boxes_xyxy_norm, confs, class_names, colors=None):
    """Draw YOLO detection boxes on a BGR image.  boxes_xyxy_norm are 0..1."""
    import cv2
    h, w = img_bgr.shape[:2]
    if colors is None:
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
    for i, (box, conf) in enumerate(zip(boxes_xyxy_norm, confs)):
        cls_id = 0  # single-class pipeline
        color = colors[cls_id % len(colors)]
        x1 = int(max(0, min(w - 1, round(box[0] * w))))
        y1 = int(max(0, min(h - 1, round(box[1] * h))))
        x2 = int(max(0, min(w - 1, round(box[2] * w))))
        y2 = int(max(0, min(h - 1, round(box[3] * h))))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = ""
        if class_names and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img_bgr


def _letterbox_np(img_rgb, target=640, color=(114, 114, 114)):
    """letterbox + return (canvas_rgb_float32_0_1, r, dw, dh, orig_h, orig_w)."""
    import cv2
    h, w = img_rgb.shape[:2]
    r = min(target / h, target / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (target - new_w) / 2.0, (target - new_h) / 2.0
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), color, dtype=np.uint8)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas.astype(np.float32) / 255.0, r, dw, dh, h, w


def _scale_boxes_back_xyxy(xyxy_pixels, r, dw, dh, orig_h, orig_w, input_sz=640):
    """Inverse of letterbox: xyxy in input canvas pixels → 0..1 normalized on orig image."""
    # pixels on canvas → subtract padding → divide by r → normalize
    x1_p, y1_p, x2_p, y2_p = xyxy_pixels.T
    x1_i = (x1_p - dw) / r
    y1_i = (y1_p - dh) / r
    x2_i = (x2_p - dw) / r
    y2_i = (y2_p - dh) / r
    return np.stack([
        np.clip(x1_i / orig_w, 0.0, 1.0),
        np.clip(y1_i / orig_h, 0.0, 1.0),
        np.clip(x2_i / orig_w, 0.0, 1.0),
        np.clip(y2_i / orig_h, 0.0, 1.0),
    ], axis=1)


def _nms_xyxy(boxes_xyxy, confs, iou_thr=0.45):
    """Very small class-agnostic NMS (numpy).  boxes_xyxy: (N,4) any coord system."""
    if len(boxes_xyxy) == 0:
        return np.zeros(0, dtype=int)
    x1 = boxes_xyxy[:, 0]; y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]; y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


# Inline worker script source — executed via SYSTEM Python (has TF installed) as
# a subprocess so we never try to import tensorflow inside the training .venv.
# The job JSON describes images + paths; this worker writes job.json.result.json
# containing raw detections (boxes+confs).  The caller (.venv with good cv2)
# handles the actual cv2.imwrite / drawing because the system Python's cv2
# build (4.13.0) is broken on this machine for numpy array drawing.
_TFL_SMOKE_WORKER_SRC = r'''
import sys, os, json, shutil
from pathlib import Path
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


def letterbox(img_rgb, target=640, color=(114,114,114)):
    # img_rgb is numpy (H,W,3) uint8
    import cv2 as _cv2
    h, w = img_rgb.shape[:2]
    r = min(target / h, target / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (target - new_w) / 2.0, (target - new_h) / 2.0
    resized = _cv2.resize(img_rgb, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), color, dtype=np.uint8)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas.astype(np.float32) / 255.0, r, dw, dh, h, w


def main(job_path):
    import cv2 as _cv2
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    tflite_src = Path(job["tflite_path"])
    ascii_dir = Path(job["ascii_dir"]); ascii_dir.mkdir(parents=True, exist_ok=True)
    conf_thr = float(job.get("conf_thresh", 0.20))

    ascii_dst = ascii_dir / "worker_smoke.tflite"
    shutil.copy2(str(tflite_src), str(ascii_dst))
    interp = tf.lite.Interpreter(model_path=str(ascii_dst))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"[worker] in={list(in_det['shape'])} {in_det['dtype'].__name__} "
          f"out={list(out_det['shape'])} {out_det['dtype'].__name__}", flush=True)

    target_sz = int(in_det["shape"][1]) if in_det["shape"][1] > 1 else 640

    per_image = []
    for img_desc in job["images"]:
        img_path = Path(img_desc["path"])
        stem = img_desc["stem"]
        name = img_path.name
        res = {"name": name, "stem": stem,
               "tflite": None,
               # Raw results for caller-side drawing:
               "_boxes_xyxy_norm": None, "_confs": None}
        try:
            img_bgr = _cv2.imread(str(img_path))
            if img_bgr is None:
                raise IOError("cv2.imread None: " + str(img_path))
            img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
            canvas, r, dw, dh, oh, ow = letterbox(img_rgb, target=target_sz)
            inp = np.expand_dims(canvas, 0)
            if in_det["dtype"] == np.uint8:
                q = in_det.get("quantization") or (1.0, 0)
                s, zp = q if q and q[0] != 0 else (1.0, 0)
                inp = np.clip((inp / s) + zp, 0, 255).astype(np.uint8)
            elif in_det["dtype"] == np.float16:
                inp = inp.astype(np.float16)
            else:
                inp = inp.astype(np.float32)
            interp.set_tensor(in_det["index"], inp)
            interp.invoke()
            raw = interp.get_tensor(out_det["index"])[0]  # (5,8400) or (8400,5)
            if raw.shape[0] == 5 and raw.shape[1] == 8400:
                raw = raw.T  # -> (8400, 5)
            if raw.shape[1] >= 4:
                xywh_pix = raw[:, :4]
                scores = raw[:, 4:].max(axis=1) if raw.shape[1] > 4 else np.zeros(raw.shape[0])
            else:
                xywh_pix = np.zeros((0,4)); scores = np.zeros(0)
            mask = scores >= conf_thr
            keep = np.where(mask)[0]
            boxes_norm = np.zeros((0,4)); cs = np.zeros(0)
            if keep.size:
                bp = xywh_pix[keep]; c_sel = scores[keep]
                x1p = bp[:,0] - bp[:,2]/2
                y1p = bp[:,1] - bp[:,3]/2
                x2p = bp[:,0] + bp[:,2]/2
                y2p = bp[:,1] + bp[:,3]/2
                xyxy_pix = np.stack([x1p,y1p,x2p,y2p], axis=1)
                # --- class-agnostic NMS (numpy, self-contained) ---
                areas = np.maximum(0, xyxy_pix[:,2]-xyxy_pix[:,0]) * np.maximum(0, xyxy_pix[:,3]-xyxy_pix[:,1])
                order = c_sel.argsort()[::-1]
                nms_keep = []
                while order.size > 0:
                    i = order[0]; nms_keep.append(int(i))
                    if order.size == 1: break
                    xx1 = np.maximum(xyxy_pix[i,0], xyxy_pix[order[1:],0])
                    yy1 = np.maximum(xyxy_pix[i,1], xyxy_pix[order[1:],1])
                    xx2 = np.minimum(xyxy_pix[i,2], xyxy_pix[order[1:],2])
                    yy2 = np.minimum(xyxy_pix[i,3], xyxy_pix[order[1:],3])
                    ww = np.maximum(0, xx2-xx1); hh = np.maximum(0, yy2-yy1)
                    inter = ww*hh
                    union = areas[i] + areas[order[1:]] - inter
                    iou = np.where(union>0, inter/union, 0.0)
                    order = order[np.where(iou<=0.45)[0] + 1]
                if len(nms_keep):
                    nk = np.array(nms_keep)
                    xyxy_pix = xyxy_pix[nk]; c_sel = c_sel[nk]
                    # Scale boxes back to 0..1 normalized on original image
                    x1i = (xyxy_pix[:,0] - dw) / r
                    y1i = (xyxy_pix[:,1] - dh) / r
                    x2i = (xyxy_pix[:,2] - dw) / r
                    y2i = (xyxy_pix[:,3] - dh) / r
                    boxes_norm = np.stack([
                        np.clip(x1i/ow, 0.0, 1.0),
                        np.clip(y1i/oh, 0.0, 1.0),
                        np.clip(x2i/ow, 0.0, 1.0),
                        np.clip(y2i/oh, 0.0, 1.0),
                    ], axis=1)
                    cs = c_sel

            top = float(cs.max()) if cs.size else 0.0
            count = int(cs.size)
            res["tflite"] = {
                "count": count, "top_conf": top,
                "confs": [round(float(c), 4) for c in cs],
            }
            res["_boxes_xyxy_norm"] = boxes_norm.tolist() if cs.size else []
            res["_confs"] = [round(float(c), 4) for c in cs]
            print(f"[worker] {name}: count={count} top={top:.4f}", flush=True)
        except Exception as e:
            import traceback
            print(f"[worker] {name}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            res["tflite"] = {"error": f"{type(e).__name__}: {e}"}
        per_image.append(res)

    resp = {"per_image": per_image}
    Path(str(job_path) + ".result.json").write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
    print("[worker] DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: worker.py <job.json>", file=sys.stderr); sys.exit(2)
    main(sys.argv[1])
'''


def run_post_train_smoke_test(pt_path, tflite_path, yaml_path, conf_thresh=0.20):
    """Smoke-test fresh PT + TFLite on every image in TEST_DIR.

    Returns a dict with {report_path, output_dir, per_image: [...], has_images: bool}.
    Failures are non-fatal: any error is logged into the report and the
    function returns with partial results instead of raising.
    """
    import cv2
    pt_path = Path(pt_path)
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
    tflite_path = Path(tflite_path) if tflite_path else None

    test_images = sorted([p for p in TEST_DIR.glob("*.jpg")] +
                         [p for p in TEST_DIR.glob("*.jpeg")] +
                         [p for p in TEST_DIR.glob("*.png")])
    result = {
        "report_path": None,
        "output_dir": None,
        "per_image": [],
        "has_images": bool(test_images),
        "pt_ok": False,
        "tflite_ok": False,
    }
    if not test_images:
        return result

    run_id = pt_path.stem.split("_")[-1] if "_" in pt_path.stem else pt_path.stem
    out_dir = OUTPUTS_DIR / f"smoke_{run_id}" if run_id else OUTPUTS_DIR / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["output_dir"] = str(out_dir)

    class_names = []
    if yaml_path and yaml_path.exists():
        try:
            class_names = read_class_names(yaml_path)
        except Exception:
            class_names = []
    if not class_names:
        class_names = [pt_path.stem.split("_")[0] if "_" in pt_path.stem else "object"]

    log_lines = [
        f"Post-train smoke test — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  PT model:    {pt_path} (exists={pt_path.exists()})",
        f"  TFLite model: {tflite_path} (exists={tflite_path.exists() if tflite_path else 'N/A'})",
        f"  YAML names:  {class_names}",
        f"  Test images: {len(test_images)} in {TEST_DIR}",
        "",
    ]

    # -------------------- PT inference (ultralytics) --------------------
    pt_results_per_img = {}
    try:
        from ultralytics import YOLO as _YOLO
        model_pt = _YOLO(str(pt_path))
        result["pt_ok"] = True
        log_lines.append("[PT] Loaded OK")
    except Exception as e:
        log_lines.append(f"[PT] Load FAILED: {type(e).__name__}: {e}")
        model_pt = None

    # -------------------- TFLite inference (Interpreter, float32) --------------------
    # NOTE: we deliberately do NOT import tensorflow in the main process, because the
    # venv used for ultralytics training typically doesn't have TF installed. Instead
    # we shell out to the system Python (py -3.x) that we know has TF/onnx2tf from the
    # post_train_export flow.  Failures here are non-fatal (TFLite section just stays
    # None); PT section above still always runs inside .venv.
    tfl_ready = {"subproc": False, "py": None}
    if tflite_path and tflite_path.exists():
        sys_py_candidates = [
            r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe",
            r"C:\Python310\python.exe",
        ]
        sys_py = None
        for cand in sys_py_candidates:
            if Path(cand).exists():
                sys_py = cand
                break
        if sys_py is None:
            # fallback: try `py -3.10 -c ""` on PATH
            import shutil as _sh_which
            py_launcher = _sh_which.which("py")
            if py_launcher:
                sys_py = py_launcher  # caller must add -3.10
                tfl_ready["py_flags"] = ["-3.10"]
            else:
                tfl_ready = {"subproc": False, "py": None}
                log_lines.append("[TFLite] Skipped: cannot locate system Python 3.10 with TensorFlow")
        if sys_py is not None:
            tfl_ready = {
                "subproc": True,
                "py": sys_py,
                "py_flags": tfl_ready.get("py_flags", []),
                "run_id": run_id,
                "tflite_path": str(tflite_path),
                "out_dir": str(out_dir),
                "class_names": class_names,
                "conf_thresh": conf_thresh,
            }
            result["tflite_ok"] = True  # optimistic; subproc will write per-result files
            log_lines.append(f"[TFLite] Will run via subprocess: {sys_py} {tfl_ready.get('py_flags', [])}")
        else:
            result["tflite_ok"] = False

    # -------------------- Per-image loop --------------------
    for img_path in test_images:
        name = img_path.name
        img_stem = img_path.stem
        log_lines.append(f"\n---- {name} ----")
        per = {"name": name, "stem": img_stem, "src_path": str(img_path), "pt": None, "tflite": None}

        # Load image once (shared)
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise IOError(f"cv2.imread returned None for {img_path}")
        except Exception as e:
            log_lines.append(f"  Image load FAILED: {type(e).__name__}: {e}")
            result["per_image"].append(per)
            continue

        # --- PT ---
        if model_pt is not None:
            try:
                res = model_pt.predict(
                    source=str(img_path),
                    conf=conf_thresh,
                    iou=0.45,
                    verbose=False,
                )[0]
                boxes_xyxy_norm = []
                confs = []
                if len(res.boxes) > 0:
                    xywhn = res.boxes.xywhn.cpu().numpy()
                    confs_arr = res.boxes.conf.cpu().numpy()
                    for xywh, c in zip(xywhn, confs_arr):
                        cx, cy, bw, bh = xywh
                        x1 = max(0.0, cx - bw / 2)
                        y1 = max(0.0, cy - bh / 2)
                        x2 = min(1.0, cx + bw / 2)
                        y2 = min(1.0, cy + bh / 2)
                        boxes_xyxy_norm.append([x1, y1, x2, y2])
                        confs.append(float(c))
                boxes_xyxy_norm = np.array(boxes_xyxy_norm).reshape(-1, 4) if boxes_xyxy_norm else np.zeros((0, 4))
                confs = np.array(confs)
                top = float(confs.max()) if confs.size else 0.0
                count = len(confs)
                per["pt"] = {"count": count, "top_conf": top, "confs": [round(float(c), 4) for c in confs]}
                log_lines.append(f"  PT:     {count} box(es), top_conf={top:.4f}")
                # Draw
                annotated_pt = img_bgr.copy()
                annotated_pt = _draw_boxes_cv2(annotated_pt, boxes_xyxy_norm, confs, class_names,
                                               colors=[(0, 200, 0)])
                pt_out = out_dir / f"{img_stem}_pt.jpg"
                cv2.imwrite(str(pt_out), annotated_pt, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                per["pt"]["out_path"] = str(pt_out)
            except Exception as e:
                log_lines.append(f"  PT inference FAILED: {type(e).__name__}: {e}")
                per["pt"] = {"error": f"{type(e).__name__}: {e}"}

        result["per_image"].append(per)

    # --- TFLite subprocess (batch all images at once) ---
    if tfl_ready.get("subproc"):
        try:
            log_lines.append("\n[TFLite] Running subprocess inference...")
            py = tfl_ready["py"]
            flags = tfl_ready.get("py_flags", [])
            # Build a JSON job description file
            job = {
                "tflite_path": tfl_ready["tflite_path"],
                "conf_thresh": tfl_ready["conf_thresh"],
                "ascii_dir": str(OUTPUTS_DIR / "tflite_ascii"),
                "images": [{"path": str(p), "stem": p.stem} for p in test_images],
            }
            job_path = TMP_DIR / f"tfl_smoke_{tfl_ready['run_id'] or 'latest'}.json"
            job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            # Subprocess script (inline) is located at TMP_DIR/_tfl_smoke_worker.py
            worker = TMP_DIR / "_tfl_smoke_worker.py"
            worker.write_text(_TFL_SMOKE_WORKER_SRC, encoding="utf-8")
            import subprocess as _sp
            cmd = [py, *flags, str(worker), str(job_path)]
            log_lines.append(f"[TFLite] cmd: {' '.join(cmd)}")
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.stdout.strip():
                log_lines.append(f"[TFLite] stdout:\n{proc.stdout.strip()}")
            if proc.returncode != 0:
                result["tflite_ok"] = False
                log_lines.append(f"[TFLite] subprocess FAILED rc={proc.returncode}")
                if proc.stderr.strip():
                    log_lines.append(f"[TFLite] stderr:\n{proc.stderr.strip()}")
            else:
                # Parse result JSON written by worker to job_path.result.json
                resp_path = Path(str(job_path) + ".result.json")
                if resp_path.exists():
                    try:
                        resp = json.loads(resp_path.read_text(encoding="utf-8"))
                    except Exception as e2:
                        log_lines.append(f"[TFLite] Failed to parse result JSON: {e2}")
                        resp = None
                    if resp:
                        # Merge raw boxes + draw TFLite annotation + composite comparison
                        # all in THIS process (.venv cv2) because system Python's cv2
                        # numpy array drawing is broken on this machine.
                        tfl_name_to_per = {
                            (per.get("name") or ""): per for per in resp.get("per_image", [])
                        }
                        for p in result["per_image"]:
                            w_per = tfl_name_to_per.get(p["name"])
                            if not w_per:
                                continue
                            p["tflite"] = w_per.get("tflite")
                            if isinstance(p["tflite"], dict) and "error" in p["tflite"]:
                                continue
                            boxes_norm = w_per.get("_boxes_xyxy_norm") or []
                            confs = w_per.get("_confs") or []
                            if not isinstance(boxes_norm, list):
                                boxes_norm = []
                            if not isinstance(confs, list):
                                confs = []
                            try:
                                img_bgr = cv2.imread(str(Path(p["src_path"])))
                                if img_bgr is None:
                                    continue
                                annotated_tfl = _draw_boxes_cv2(
                                    img_bgr.copy(),
                                    np.array(boxes_norm, dtype=np.float32) if len(boxes_norm) else np.zeros((0,4), dtype=np.float32),
                                    np.array(confs, dtype=np.float32) if len(confs) else np.zeros(0, dtype=np.float32),
                                    class_names=class_names,
                                    colors=[(0, 128, 255)])
                                tfl_out = out_dir / f"{p['stem']}_tflite.jpg"
                                cv2.imwrite(str(tfl_out), annotated_tfl, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                                if isinstance(p.get("tflite"), dict):
                                    p["tflite"]["out_path"] = str(tfl_out)
                                # Now build the PT vs TFLite side-by-side composite
                                pt_path = p.get("pt", {}).get("out_path") if isinstance(p.get("pt"), dict) else None
                                if pt_path and Path(pt_path).exists():
                                    a = cv2.imread(str(pt_path))
                                    b = cv2.imread(str(tfl_out))
                                    if a is not None and b is not None:
                                        hm = max(a.shape[0], b.shape[0])
                                        def fit(im, H):
                                            if im.shape[0] == H:
                                                return im
                                            r = H / im.shape[0]
                                            return cv2.resize(im, (int(im.shape[1]*r), H), interpolation=cv2.INTER_LINEAR)
                                        a2 = fit(a, hm)
                                        b2 = fit(b, hm)
                                        pt_info = p.get("pt", {})
                                        tfl_info = p.get("tflite", {}) if isinstance(p.get("tflite"), dict) else {}
                                        pt_count = pt_info.get("count", 0) if isinstance(pt_info, dict) else 0
                                        pt_top = pt_info.get("top_conf", 0.0) if isinstance(pt_info, dict) else 0.0
                                        tf_count = tfl_info.get("count", 0) if isinstance(tfl_info, dict) else 0
                                        tf_top = tfl_info.get("top_conf", 0.0) if isinstance(tfl_info, dict) else 0
                                        def cap(im, txt):
                                            hh, ww = im.shape[:2]
                                            c = np.full((40, ww, 3), (240,240,240), dtype=np.uint8)
                                            cv2.putText(c, txt, (10, 28),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)
                                            return np.concatenate([c, im], axis=0)
                                        a3 = cap(a2, f"PyTorch PT  (boxes={pt_count}, top={pt_top:.3f})")
                                        b3 = cap(b2, f"TFLite      (boxes={tf_count}, top={tf_top:.3f})")
                                        combo = np.concatenate([a3, b3], axis=1)
                                        combo_out = out_dir / f"{p['stem']}_compare.jpg"
                                        cv2.imwrite(str(combo_out), combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                                        p["compare_path"] = str(combo_out)
                            except Exception as e_draw:
                                log_lines.append(f"[TFLite] Draw error for {p['name']}: {type(e_draw).__name__}: {e_draw}")
                        tf_hits_count = 0
                        for p in result["per_image"]:
                            tf = p.get("tflite") or {}
                            if isinstance(tf, dict) and "error" not in tf and tf.get("count", 0) > 0:
                                tf_hits_count += 1
                        result["tflite_hits"] = tf_hits_count
                        if "error" not in resp:
                            log_lines.append(f"[TFLite] Subprocess completed OK ({tf_hits_count}/{len(test_images)} hits)")
        except Exception as e:
            log_lines.append(f"[TFLite] Subprocess orchestration FAILED: {type(e).__name__}: {e}")
            import traceback
            log_lines.append("    " + "\n    ".join(traceback.format_exc().splitlines()[-5:]))
            result["tflite_ok"] = False

    # ---- summary block ----
    log_lines.append("\n==== SUMMARY ====")
    pt_ok_n = sum(1 for p in result["per_image"]
                  if isinstance(p.get("pt"), dict) and "error" not in p["pt"] and p["pt"].get("count", 0) > 0)
    if "tflite_hits" not in result:
        tf_ok_n = sum(1 for p in result["per_image"]
                      if isinstance(p.get("tflite"), dict) and "error" not in p["tflite"] and p["tflite"].get("count", 0) > 0)
    else:
        tf_ok_n = result["tflite_hits"]
    log_lines.append(f"  PT hits:     {pt_ok_n}/{len(test_images)} images had >=1 box above {conf_thresh}")
    log_lines.append(f"  TFLite hits: {tf_ok_n}/{len(test_images)} images had >=1 box above {conf_thresh}")
    result["pt_hits"] = pt_ok_n
    result["tflite_hits"] = tf_ok_n
    result["total_images"] = len(test_images)

    report_path = out_dir / f"smoke_report_{run_id or 'latest'}.txt"
    try:
        report_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        result["report_path"] = str(report_path)
    except Exception:
        pass
    return result


def run_pipeline(video_path, class_name, config):
    try:
        reset_state()
        video_path = Path(video_path)

        fps = config.get("fps", 2)
        epochs = config.get("epochs", 100)
        imgsz = config.get("imgsz", 640)
        batch = config.get("batch", 8)
        device = config.get("device", "0")
        workers = config.get("workers", 0)
        train_ratio = config.get("train_ratio", 0.9)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_run_dir = DATASETS_DIR / run_id
        frames_dir = EXTRACTIONS_DIR / run_id
        labels_dir = dataset_run_dir / "labels"
        images_dir = dataset_run_dir / "images"
        dataset_dir = dataset_run_dir / "dataset"

        for d in [frames_dir, labels_dir, images_dir, dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)

        update_status("extracting", 0, "Extracting frames from video...", f"Starting frame extraction (fps={fps})")
        frame_count, width, height = extract_frames(video_path, frames_dir, fps=fps)
        update_status("extracting", 100, f"Extracted {frame_count} frames", f"Extracted {frame_count} frames ({width}x{height})")

        update_status("labeling", 0, "Auto-labeling frames with LLM...", "Starting auto-labeling: LLM returns boxes per frame")
        labeled_count = auto_label_frames(frames_dir, labels_dir, class_name)
        update_status("labeling", 100, f"Labeled {labeled_count}/{frame_count} frames", f"Labeling complete: {labeled_count}/{frame_count} frames with detections")

        # --------------------------- MANUAL LABEL BRIDGE ---------------------------
        # If the user checked "允许人工标注补漏" (default True), pause the pipeline
        # thread here after auto-labeling and expose the frames that have NO .txt label
        # yet (either LLM returned boxes=[], LLM failed and aborted early, or any
        # other gap) to the web UI for manual mouse-drag bbox annotation.
        # After user presses "完成", /api/manual_label/complete calls
        # _MANUAL_LABEL_EVENT.set() which returns us to splitting/training below.
        enable_manual_label = bool(config.get("enable_manual_label", True))
        if enable_manual_label:
            all_jpgs = sorted([p.name for p in frames_dir.glob("*.jpg")])
            if all_jpgs:
                # Unlabeled = JPG exists but corresponding YOLO .txt does NOT yet exist.
                # This covers every case:
                #   - LLM returned "boxes":[] → we never write .txt (matches skip semantics).
                #   - LLM crashed mid-run (5-frame empty / bad-json abort) → tail frames have no txt.
                #   - LLM genuinely couldn't find the object in those frames → user chooses
                #     "跳过此帧" which writes an empty .txt (negative / background sample).
                unlabeled_basenames = [
                    jpg_name
                    for jpg_name in all_jpgs
                    if not (labels_dir / (Path(jpg_name).stem + ".txt")).exists()
                ]
                # --- UX fix: even when LLM labeled *everything* (unlabeled=[]),
                # we still hang at waiting_manual_label so the user gets clear
                # feedback ("全标成功") instead of silently jumping to training.
                # Two buttons are then exposed in the UI:
                #   (a) "直接训练（不复核）" → /api/manual_label/complete → Event.set()
                #   (b) "复核全部帧（编辑 LLM 已有框）" → /api/manual_label/review_all
                #       → todo becomes all_jpgs, pre_saved_boxes loaded from existing .txt
                # We also need a way to know which mode we're in, so stash it in _MANUAL_LABEL_CTX.
                if unlabeled_basenames:
                    initial_todo = list(unlabeled_basenames)
                    mode_hint = "patch"  # patch missing labels only
                    user_msg = (
                        f"LLM 标注完成，待人工补标 {len(unlabeled_basenames)}/{len(all_jpgs)} 张。"
                        f" 请在页面框选后点『全部完成』继续；也可切换到『复核全部帧』模式检查全部图片。"
                    )
                    detail_msg = (
                        f"Manual labeling phase (PATCH mode): "
                        f"{len(unlabeled_basenames)} unlabeled / {len(all_jpgs)} total frames. "
                        f"UI shows each frame without prior bboxes; user can switch to REVIEW mode "
                        f"to see & edit LLM-produced bboxes on every frame."
                    )
                else:
                    initial_todo = list(all_jpgs)  # will show pre-saved boxes; user picks review-or-skip
                    mode_hint = "review_prompt"    # shows two action buttons instead of canvas first
                    user_msg = (
                        f"🎉 LLM 已成功标注 {len(all_jpgs)} 张（全部成功，0 张遗漏）。"
                        f" 请选择下一步：『直接训练』跳过人工复核，或『开始复核全部帧』手动检查/编辑 LLM 结果。"
                    )
                    detail_msg = (
                        f"Manual labeling phase (REVIEW prompt): all {len(all_jpgs)} frames already "
                        f"have YOLO .txt from LLM. UI presents two actions: (1) skip manual entirely, "
                        f"(2) enter REVIEW mode which loads the LLM-produced bboxes onto canvas so "
                        f"user can delete/adjust/add boxes before committing to training."
                    )
                _MANUAL_LABEL_EVENT.clear()
                _MANUAL_LABEL_CTX.update({
                    "run_id": run_id,
                    "frames_dir": str(frames_dir),
                    "labels_dir": str(labels_dir),
                    "todo_filenames": list(initial_todo),
                    "done_filenames": set(),
                    "all_filenames": list(all_jpgs),
                    "mode": mode_hint,   # "patch" | "review_prompt" | "review"
                })
                update_status(
                    "waiting_manual_label",
                    50,
                    user_msg,
                    detail_msg,
                )
                # Block the pipeline daemon thread indefinitely here. This is why
                # we run run_pipeline in a threading.Thread (not the Flask request
                # handler): a synchronous HTTP worker would tie up a gunicorn/werkzeug
                # worker indefinitely. /api/reset clears the event via reset_state()
                # above so a canceled run can be restarted cleanly without deadlock.
                _MANUAL_LABEL_EVENT.wait()
                # Count txts again after manual save/skip pass to propagate correct
                # "labeled_count" to set_done() result + "if labeled_count == 0" check.
                labeled_count = len(list(labels_dir.glob("*.txt")))

        if labeled_count == 0:
            raise RuntimeError(
                "No objects detected in any frame. Please check:\n"
                "  1) LLM (http://127.0.0.1:12345) 已加载视觉模型（qwen3.5-VL 等）\n"
                "  2) 输入的描述文字清晰说明要追踪的物品（颜色/材质/形状等特征）\n"
                "  3) 视频中目标物品清晰可见，且大多数帧中存在"
            )

        update_status("splitting", 0, "Splitting dataset into train/val...", f"Train ratio: {train_ratio}")
        split_dataset(frames_dir, labels_dir, train_ratio=train_ratio)

        for d in [dataset_dir / "images", dataset_dir / "labels"]:
            if d.exists():
                shutil.rmtree(str(d))

        shutil.copytree(str(frames_dir / "train"), str(dataset_dir / "images" / "train"))
        shutil.copytree(str(frames_dir / "val"), str(dataset_dir / "images" / "val"))
        shutil.copytree(str(labels_dir / "train"), str(dataset_dir / "labels" / "train"))
        shutil.copytree(str(labels_dir / "val"), str(dataset_dir / "labels" / "val"))

        yaml_path = dataset_run_dir / "dataset.yaml"
        generate_yaml(dataset_dir, class_name, yaml_path)
        update_status("splitting", 100, "Dataset split complete", f"YAML config: {yaml_path}")

        update_status("training", 0, f"Training YOLO model (epochs={epochs})...", "Starting YOLO training")

        # Load base checkpoint from the canonical weights folder; if it is missing fall back
        # to the bare filename so ultralytics will auto-download the default into cwd.
        base_weights = WEIGHTS_DIR / "yolo11n.pt"
        model_arg = str(base_weights) if base_weights.exists() else "yolo11n.pt"
        model = YOLO(model_arg)

        progress_callback = []

        class ProgressTracker:
            def __init__(self):
                self.last_pct = -1

            def __call__(self, trainer):
                try:
                    pct = int(trainer.epoch / trainer.epochs * 100)
                    if pct != self.last_pct:
                        self.last_pct = pct
                        update_status(
                            "training",
                            pct,
                            f"Training epoch {trainer.epoch}/{trainer.epochs}",
                            f"Epoch {trainer.epoch}/{trainer.epochs}",
                        )
                except Exception:
                    pass

        tracker = ProgressTracker()

        # Register progress via model.add_callback. The older `callbacks={...}` kwarg was
        # removed from ultralytics get_cfg in recent versions (causes
        # SyntaxError: 'callbacks' is not a valid YOLO argument), so we must NOT pass it
        # through the overrides dict (which is what **model.train(kwargs) feeds to get_cfg).
        # See https://docs.ultralytics.com/integrations/callbacks/
        try:
            model.add_callback("on_fit_epoch_end", tracker)
        except Exception:
            # Some older builds still use the `on_epoch_end` event name instead
            try:
                model.add_callback("on_epoch_end", tracker)
            except Exception:
                # If neither event works we still want to run training; progress bar will
                # just stay coarse-grained from the run_pipeline finalization markers.
                pass

        train_kwargs = dict(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(OUTPUTS_DIR),
            name=run_id,
            # NOTE: do NOT add `callbacks=...` here. Ultralytics >= v8.4 rejects it with
            #       SyntaxError: 'callbacks' is not a valid YOLO argument.
        )
        results = model.train(**train_kwargs)

        best_model_path = OUTPUTS_DIR / run_id / "weights" / "best.pt"
        if not best_model_path.exists():
            best_model_path = OUTPUTS_DIR / run_id / "weights" / "last.pt"

        final_model_path = OUTPUTS_DIR / f"{class_name}_{run_id}.pt"
        shutil.copy2(str(best_model_path), str(final_model_path))

        # ------------------------------------------------------------------
        # Auto-export pt -> tflite + labels.txt for immediate Android use.
        # Export failures do NOT invalidate training (pt is still valid); we
        # just annotate the status/result so the UI can surface them.
        # ------------------------------------------------------------------
        update_status(
            "exporting", 95,
            "Training complete. Converting .pt -> TFLite for Android...",
            f"Starting post-train export: {final_model_path.name} -> detect.tflite + labels.txt",
        )
        android_assets = BASE_DIR.parent / "android-app" / "app" / "src" / "main" / "assets"
        try:
            export_ok, tflite_path, labels_path, copied_android = post_train_export(
                pt_path=final_model_path,
                yaml_path=yaml_path,
                output_dir=OUTPUTS_DIR,
                android_assets_dir=android_assets,
                imgsz=imgsz,
            )
        except Exception as e:
            export_ok, tflite_path, labels_path, copied_android = False, None, None, False
            update_status("done", 100, "Training OK, but export failed.", f"post_train_export crashed: {type(e).__name__}: {e}")
        else:
            if export_ok:
                msg = "Training + export complete!"
                detail_bits = [f"Model saved: {final_model_path.name}", f"TFLite: {tflite_path.name if tflite_path else 'N/A'}"]
                detail_bits.append(f"Labels: {labels_path.name if labels_path else 'N/A'}")
                if copied_android:
                    detail_bits.append("Auto-copied to android-app assets (ready for gradlew build)")
                update_status("done", 97, msg, " | ".join(detail_bits))
            else:
                update_status(
                    "done", 97,
                    "Training OK, export partially failed. Check export_report txt.",
                    f"pt model saved OK ({final_model_path.name}); tflite export had errors — see outputs/export_report_{final_model_path.stem}.txt",
                )

        # ------------------------------------------------------------------
        # Post-train smoke test on real-world images in ml/test/ (if any).
        # Failures here are 100% non-fatal — we just add the result to the
        # overall status/result so the user can review.
        # ------------------------------------------------------------------
        smoke_result = None
        try:
            update_status(
                "smoke_testing", 98,
                "Running smoke test on ml/test/ images...",
                "Running PT + TFLite inference on user-provided test images",
            )
            conf_thresh = float(config.get("conf_threshold", 0.20))
            smoke_result = run_post_train_smoke_test(
                pt_path=final_model_path,
                tflite_path=tflite_path,
                yaml_path=yaml_path,
                conf_thresh=conf_thresh,
            )
            if smoke_result.get("has_images"):
                pt_hits = smoke_result.get("pt_hits", 0)
                tf_hits = smoke_result.get("tflite_hits", 0)
                total = smoke_result.get("total_images", 0)
                out_dir = smoke_result.get("output_dir") or ""
                rep = smoke_result.get("report_path") or ""
                msg_smoke = (
                    f"Smoke test: PT {pt_hits}/{total}, TFLite {tf_hits}/{total} "
                    f"(threshold {conf_thresh})."
                )
                detail_smoke = (
                    f"Outputs in {out_dir}; report {rep}"
                )
                update_status("done", 100, msg_smoke, detail_smoke)
            else:
                # no test images → just keep existing done status at 100
                pass
        except Exception as e:
            # never let smoke test bring down the overall pipeline
            smoke_result = {"error": f"{type(e).__name__}: {e}"}
            import traceback
            traceback.print_exc()

        done_payload = {
            "model_path": str(final_model_path),
            "model_name": final_model_path.name,
            "tflite_path": str(tflite_path) if tflite_path and tflite_path.exists() else None,
            "labels_path": str(labels_path) if labels_path and labels_path.exists() else None,
            "export_ok": bool(export_ok),
            "copied_to_android_assets": bool(copied_android),
            "run_id": run_id,
            "frames": frame_count,
            "labeled": labeled_count,
            "class_name": class_name,
            "smoke_test": smoke_result,
        }
        set_done(done_payload)

    except Exception as e:
        set_error(str(e))
        import traceback
        traceback.print_exc()


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def api_status():
    return jsonify(get_status())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}. Allowed: {', '.join(allowed)}"}), 400

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = UPLOADS_DIR / f"{run_id}{ext}"
    f.save(str(video_path))

    return jsonify({"video_path": str(video_path), "filename": f.filename})


@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json()
    video_path = data.get("video_path")
    class_name = data.get("class_name", "object")

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 400

    class_name = class_name.strip()
    if not class_name:
        return jsonify({"error": "Class name is required"}), 400

    config = {
        "fps": data.get("fps", 2),
        "epochs": data.get("epochs", 100),
        "imgsz": data.get("imgsz", 640),
        "batch": data.get("batch", 8),
        "device": data.get("device", "0"),
        "workers": data.get("workers", 0),
        "conf_threshold": data.get("conf_threshold", 0.25),
        "train_ratio": data.get("train_ratio", 0.9),
        "use_llm": data.get("use_llm", False),
        "enable_manual_label": bool(data.get("enable_manual_label", True)),
    }

    t = threading.Thread(target=run_pipeline, args=(video_path, class_name, config), daemon=True)
    t.start()

    return jsonify({"status": "started"})


@app.route("/api/download/<path:filename>")
def api_download(filename):
    safe_name = os.path.basename(filename)
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=safe_name)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_state()
    return jsonify({"status": "reset"})


# -------------------------- Manual Labeling APIs --------------------------
# These APIs form the bidirectional bridge between the web UI's "人工补标"
# canvas widget and the run_pipeline daemon thread (blocked on
# _MANUAL_LABEL_EVENT.wait() inside run_pipeline MANUAL LABEL BRIDGE above).

@app.route("/api/frame/<run_id>/<path:filename>")
def api_manual_frame(run_id: str, filename: str):
    """Serve extracted raw JPG for a single frame inside the manual-label phase.

    Security: only allow serving files directly under EXTRACTIONS_DIR/<run_id>/
    (canonical per-run frame directory).  Path traversal (../../) is blocked by
    stripping to just the leaf basename of `filename` before joining.
    """
    safe_basename = Path(filename).name  # drop any directory components
    candidate = EXTRACTIONS_DIR / run_id / safe_basename
    # Double-check the resolved path still lives under EXTRACTIONS_DIR after symlink follow
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return jsonify({"error": "Frame not found"}), 404
    if not str(resolved).startswith(str(EXTRACTIONS_DIR.resolve())):
        return jsonify({"error": "Forbidden"}), 403
    return send_file(str(resolved))


@app.route("/api/manual_label/todo")
def api_manual_todo():
    """Return the list of frames still missing a label + existing LLM-produced bboxes.

    Called by the UI as soon as /api/status reports step='waiting_manual_label'.
    Response shape:
      { run_id, mode, todo, done, total, all_count, unlabeled_count, labeled_count,
        pre_saved_boxes: { filename: [{x1,y1,x2,y2}] } }

    pre_saved_boxes contains one entry per frame that already has a YOLO .txt in
    labels_dir. Coordinates are 0..1 normalized, converted from YOLO
    class_id cx cy bw bh → JS expects x1y1x2y2 (two-point rectangle) so frontend
    can load them onto the canvas for editing / deletion / adjustment.
    """
    def _read_yolo_boxes(label_path: Path):
        """Return list of {x1,y1,x2,y2} dicts (0..1 normalized) or [] if file missing/empty.
        """
        if not label_path.exists():
            return []
        try:
            text = label_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        result = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                # class_id parts[0] is ignored here (frontend always uses class 0 for single-class pipeline)
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                continue
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(1.0, cx + bw / 2.0)
            y2 = min(1.0, cy + bh / 2.0)
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            result.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return result

    with state_lock:
        todo = list(_MANUAL_LABEL_CTX["todo_filenames"])
        done = sorted(_MANUAL_LABEL_CTX["done_filenames"])
        mode = _MANUAL_LABEL_CTX.get("mode", "patch")
        all_files = _MANUAL_LABEL_CTX.get("all_filenames") or list(todo)
        labels_dir = Path(_MANUAL_LABEL_CTX["labels_dir"]) if _MANUAL_LABEL_CTX["labels_dir"] else None

    # Counts (always compute outside lock – pure filesystem reads, cheap & consistent)
    all_count = len(all_files)
    pre_saved_boxes = {}
    labeled_count = 0
    if labels_dir is not None and labels_dir.exists():
        for jpg_name in all_files:
            stem = Path(jpg_name).stem
            lbl = labels_dir / f"{stem}.txt"
            if lbl.exists():
                labeled_count += 1
                boxes = _read_yolo_boxes(lbl)
                if boxes:
                    pre_saved_boxes[jpg_name] = boxes
        for jpg_name in todo:
            if jpg_name in pre_saved_boxes:
                continue  # already loaded via all_files sweep above
            stem = Path(jpg_name).stem
            lbl = labels_dir / f"{stem}.txt"
            if lbl.exists():
                boxes = _read_yolo_boxes(lbl)
                if boxes:
                    pre_saved_boxes[jpg_name] = boxes
    unlabeled_count = max(0, all_count - labeled_count)

    return jsonify({
        "run_id": _MANUAL_LABEL_CTX["run_id"],
        "mode": mode,
        "todo": todo,
        "done": done,
        "total": len(todo),
        "all_count": all_count,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "pre_saved_boxes": pre_saved_boxes,
    })


@app.route("/api/manual_label/review_all", methods=["POST"])
def api_manual_review_all():
    """Switch manual-label session into REVIEW mode:

      * todo list becomes ALL frames (not just unlabeled ones)
      * mode is reset to "review" so the UI shows pre_saved_boxes on canvas
      * done_filenames is cleared so user walks through every frame once

    Callable from any mode:
      - review_prompt (LLM labeled everything): user clicked "开始复核全部帧"
      - patch (LLM missed some frames): user clicked "切换到复核全部帧" in toolbar

    Always redirects to regular canvas UI; caller should re-GET /api/manual_label/todo.
    """
    if not _MANUAL_LABEL_CTX["run_id"]:
        return jsonify({"error": "No active manual-label session. Start training first."}), 400
    with state_lock:
        all_files = _MANUAL_LABEL_CTX.get("all_filenames") or list(_MANUAL_LABEL_CTX["todo_filenames"])
        _MANUAL_LABEL_CTX["todo_filenames"] = list(all_files)
        _MANUAL_LABEL_CTX["done_filenames"] = set()
        _MANUAL_LABEL_CTX["mode"] = "review"
    return jsonify({
        "ok": True,
        "mode": "review",
        "total": len(all_files),
        "all_count": len(all_files),
    })


@app.route("/api/manual_label/save", methods=["POST"])
def api_manual_save():
    """Persist a list of user-drawn bboxes into YOLO .txt label for one frame.

    Input JSON: { filename: "frame_0007.jpg", boxes: [[cx,cy,bw,bh], ...] }
      - coords are 0..1 normalized (same format YOLO dataset expects).
      - class_id is always written as 0 (single-class pipeline, matches
        dataset.yaml names=[class_name] → index 0).
    """
    body = request.get_json(force=True, silent=True) or {}
    filename = str(body.get("filename", "")).strip()
    boxes = body.get("boxes") or []
    if not filename or not _MANUAL_LABEL_CTX["run_id"]:
        return jsonify({"error": "Missing filename / no active manual-label session"}), 400
    if not isinstance(boxes, list):
        return jsonify({"error": "'boxes' must be a list of [cx,cy,bw,bh] normalized coords"}), 400

    safe_base = Path(filename).name
    labels_dir = Path(_MANUAL_LABEL_CTX["labels_dir"])
    if not labels_dir.exists():
        return jsonify({"error": "Labels directory not ready yet"}), 400
    label_path = labels_dir / f"{Path(safe_base).stem}.txt"

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    lines_out = []
    for box in boxes:
        if not (isinstance(box, (list, tuple)) and len(box) >= 4):
            continue
        try:
            cx, cy, bw, bh = _clamp(box[0]), _clamp(box[1]), _clamp(box[2]), _clamp(box[3])
        except (TypeError, ValueError):
            continue
        if bw < 1e-4 or bh < 1e-4:
            continue  # ignore zero-size box (occasional 1-px click-drag glitch)
        lines_out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    label_path.write_text("".join(lines_out), encoding="utf-8")
    # Mark frame as "processed" so the UI won't re-surface it when reloading
    # the todo list after save.  We write to done_filenames regardless of
    # whether lines_out is empty: user pressed "Save" explicitly on a 0-box
    # canvas → they considered it a negative / background frame (which is what
    # an empty .txt means to both YOLO and our split_dataset copytree below).
    with state_lock:
        _MANUAL_LABEL_CTX["done_filenames"].add(safe_base)
    return jsonify({
        "ok": True,
        "saved": label_path.name,
        "box_count": len(lines_out),
    })


@app.route("/api/manual_label/skip", methods=["POST"])
def api_manual_skip():
    """Mark frame as "no target present" by writing an empty YOLO label.

    Semantically equivalent to save() with boxes=[] (which also writes an
    empty .txt). This gives the UI a clearly-labelled action path so users
    don't have to think about "Save with zero boxes = negative sample".

    An empty .txt tells both split_dataset (copytree into train split) and
    ultralytics YOLO: "this frame contains zero instances of any class"
    → treated as negative / background sample, improves classifier margin.
    """
    body = request.get_json(force=True, silent=True) or {}
    filename = str(body.get("filename", "")).strip()
    if not filename or not _MANUAL_LABEL_CTX["run_id"]:
        return jsonify({"error": "Missing filename / no active manual-label session"}), 400
    safe_base = Path(filename).name
    labels_dir = Path(_MANUAL_LABEL_CTX["labels_dir"])
    label_path = labels_dir / f"{Path(safe_base).stem}.txt"
    label_path.write_text("", encoding="utf-8")  # empty = negative frame
    with state_lock:
        _MANUAL_LABEL_CTX["done_filenames"].add(safe_base)
    return jsonify({"ok": True, "skipped": label_path.name})


@app.route("/api/manual_label/complete", methods=["POST"])
def api_manual_complete():
    """Release the run_pipeline daemon thread to continue with dataset split
    and training phases once the user has finished save/skip on every frame
    they care about (or explicitly choose to proceed with remaining gaps).

    We do NOT force the user to touch every frame: we accept "complete" even
    when some todo items are not in done_filenames, because those untouched
    frames simply keep their original "no .txt" state → behave the same as
    they would without the manual-label phase at all (equivalent to cancelling
    the manual pass on those specific frames).
    """
    if not _MANUAL_LABEL_CTX["run_id"]:
        return jsonify({"error": "No active manual-label session. Start training first."}), 400
    remaining = sum(
        1 for f in _MANUAL_LABEL_CTX["todo_filenames"]
        if f not in _MANUAL_LABEL_CTX["done_filenames"]
    )
    # Signal the daemon thread (blocked on _MANUAL_LABEL_EVENT.wait()) to
    # re-scan the labels directory and proceed to splitting + training.
    _MANUAL_LABEL_EVENT.set()
    update_log_line = (
        f"Manual labeling complete signaled by user. "
        f"{len(_MANUAL_LABEL_CTX['todo_filenames']) - remaining}/{len(_MANUAL_LABEL_CTX['todo_filenames'])} frames saved/skipped; "
        f"{remaining} frames left untouched (no .txt → will remain unlabeled same as pre-manual-pass)."
    )
    update_status(
        "waiting_manual_label",
        50,
        f"人工标注结束，继续训练流程（{remaining} 帧未处理，视为未标注）。",
        update_log_line,
    )
    return jsonify({
        "ok": True,
        "processed": len(_MANUAL_LABEL_CTX["todo_filenames"]) - remaining,
        "total": len(_MANUAL_LABEL_CTX["todo_filenames"]),
        "untouched": remaining,
    })


@app.route("/api/llm_status")
def api_llm_status():
    available = llm_available()
    return jsonify({
        "available": available,
        "url": LLM_BASE_URL if available else None,
        "model": LLM_MODEL if available else None,
    })


HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO 视频训练平台</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
    color: #e0e0e0;
    padding: 20px;
  }
  .container { max-width: 960px; margin: 0 auto; }
  h1 {
    text-align: center;
    font-size: 2rem;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle {
    text-align: center;
    color: #999;
    margin-bottom: 30px;
    font-size: 0.95rem;
  }
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 16px; color: #b8b8ff; }

  .upload-area {
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    background: rgba(255,255,255,0.02);
  }
  .upload-area:hover, .upload-area.dragover {
    border-color: #667eea;
    background: rgba(102,126,234,0.1);
  }
  .upload-area .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-area p { color: #aaa; }
  .upload-area .filename { color: #667eea; font-weight: 500; margin-top: 8px; }
  input[type="file"] { display: none; }

  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; margin-bottom: 6px; color: #bbb; font-size: 0.9rem; }
  .form-group input, .form-group select {
    width: 100%;
    padding: 10px 14px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: #fff;
    font-size: 0.95rem;
    transition: border-color 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: #667eea;
  }
  .form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }

  .btn {
    display: inline-block;
    padding: 12px 28px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
  }
  .btn-primary {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
  }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary {
    background: rgba(255,255,255,0.1);
    color: #ddd;
    border: 1px solid rgba(255,255,255,0.2);
  }
  .btn-secondary:hover:not(:disabled) { background: rgba(255,255,255,0.15); }
  .btn-group { display: flex; gap: 12px; margin-top: 20px; }

  .progress-section { display: none; }
  .progress-section.active { display: block; }
  .progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 4px;
    transition: width 0.3s;
    width: 0%;
  }
  .progress-info { display: flex; justify-content: space-between; font-size: 0.85rem; color: #aaa; }
  .step-indicator {
    display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;
  }
  .step {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    background: rgba(255,255,255,0.05);
    color: #888;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .step.active { background: rgba(102,126,234,0.3); color: #fff; border-color: #667eea; }
  .step.done { background: rgba(76,175,80,0.2); color: #81c784; border-color: #4caf50; }

  .log-area {
    background: rgba(0,0,0,0.4);
    border-radius: 8px;
    padding: 14px;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 0.8rem;
    max-height: 250px;
    overflow-y: auto;
    margin-top: 12px;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .log-area .log-line { color: #90caf9; margin-bottom: 2px; }
  .log-area .log-line.error { color: #ef5350; }
  .log-area .log-line.success { color: #66bb6a; }

  .result-section { display: none; }
  .result-section.active { display: block; }
  .result-box {
    background: rgba(76,175,80,0.1);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }
  .result-box .icon { font-size: 3rem; margin-bottom: 12px; }
  .result-box h3 { color: #81c784; margin-bottom: 8px; }
  .result-box p { color: #aaa; margin-bottom: 16px; }
  .download-btn {
    display: inline-block;
    padding: 12px 32px;
    background: linear-gradient(90deg, #4caf50, #2e7d32);
    color: white;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s;
  }
  .download-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(76,175,80,0.4); }

  .settings-toggle {
    display: flex; align-items: center; gap: 8px;
    cursor: pointer; color: #999; font-size: 0.85rem;
    user-select: none;
  }
  .settings-toggle:hover { color: #ccc; }
  .settings-content { display: none; margin-top: 16px; }
  .settings-content.open { display: block; }

  .status-badge {
    display: inline-block; padding: 4px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
  }
  .status-idle { background: rgba(255,255,255,0.1); color: #aaa; }
  .status-running { background: rgba(102,126,234,0.3); color: #b8b8ff; }
  .status-done { background: rgba(76,175,80,0.3); color: #81c784; }
  .status-error { background: rgba(244,67,54,0.3); color: #ef5350; }

  /* Manual Labeling Section */
  .manual-section { display: none; }
  .manual-section.active { display: block; }
  .manual-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; flex-wrap: wrap; gap: 10px;
  }
  .manual-title { font-size: 1rem; color: #b8b8ff; font-weight: 600; }
  .manual-counter {
    font-size: 0.85rem; padding: 5px 12px; border-radius: 14px;
    background: rgba(102,126,234,0.2); color: #a5b4fc;
  }
  .canvas-wrap {
    background: rgba(0,0,0,0.4); border-radius: 10px; padding: 10px;
    display: flex; justify-content: center; align-items: center;
    border: 1px solid rgba(255,255,255,0.08); overflow: auto;
    max-height: 520px;
  }
  #labelCanvas {
    display: block; max-width: 100%; cursor: crosshair;
    border-radius: 6px; background: #1a1a2e;
  }
  .manual-hint {
    font-size: 0.8rem; color: #888; margin-top: 10px;
    padding: 10px 12px; background: rgba(0,0,0,0.25);
    border-radius: 8px; line-height: 1.6;
  }
  .manual-hint code {
    background: rgba(255,255,255,0.1); padding: 2px 6px;
    border-radius: 4px; color: #a5b4fc; font-size: 0.78rem;
  }
  .box-list {
    margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;
  }
  .box-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 14px;
    background: rgba(102,126,234,0.15); color: #a5b4fc;
    font-size: 0.78rem; border: 1px solid rgba(102,126,234,0.3);
  }
  .box-chip .x-btn {
    cursor: pointer; color: #ef5350; font-weight: 700;
    padding: 0 2px; line-height: 1;
  }
  .box-chip .x-btn:hover { color: #ff8a80; }
  .empty-boxes { color: #888; font-size: 0.8rem; font-style: italic; }

  /* Review-prompt stats cards (shown when LLM labeled 100% of frames) */
  .rp-stat {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 14px 22px;
    min-width: 140px;
    text-align: center;
  }
  .rp-num {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
  }
  .rp-label {
    font-size: 0.8rem;
    opacity: 0.75;
  }
  .manual-mode-label {
    user-select: none;
  }
</style>
</head>
<body>
<div class="container">
  <h1>🎯 YOLO 视频训练平台</h1>
  <p class="subtitle">上传视频 → 自动标注 → 训练模型 → 输出 .pt</p>

  <div class="card">
    <h2>1. 上传视频</h2>
    <div class="upload-area" id="uploadArea">
      <div class="icon">📹</div>
      <p>点击选择视频文件，或拖拽到此处</p>
      <p style="font-size:0.8rem;color:#666;margin-top:8px;">支持 MP4, AVI, MOV, MKV, WEBM 等格式</p>
      <div class="filename" id="fileName"></div>
    </div>
    <input type="file" id="videoFile" accept="video/*">
  </div>

  <div class="card">
    <h2>2. 配置识别目标</h2>
    <div class="form-group">
      <label>识别目标（用英文，如 beibingyang_can、bottle、red_car）</label>
      <input type="text" id="className" placeholder="例如：beibingyang_can" value="beibingyang_can">
    </div>

    <div class="settings-toggle" onclick="toggleSettings()">
      <span>⚙️ 高级设置（点击展开）</span>
    </div>
    <div class="settings-content" id="settingsContent">
      <div class="form-row">
        <div class="form-group">
          <label>抽帧频率 (FPS)</label>
          <input type="number" id="fps" value="2" min="0.5" max="30" step="0.5">
        </div>
        <div class="form-group">
          <label>训练/验证比例</label>
          <input type="number" id="trainRatio" value="0.9" min="0.5" max="0.95" step="0.05">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>训练轮数 (Epochs)</label>
          <input type="number" id="epochs" value="100" min="1" max="1000">
        </div>
        <div class="form-group">
          <label>图像尺寸</label>
          <select id="imgsz">
            <option value="640" selected>640</option>
            <option value="480">480</option>
            <option value="320">320</option>
            <option value="1280">1280</option>
          </select>
        </div>
        <div class="form-group">
          <label>批次大小 (Batch)</label>
          <input type="number" id="batch" value="8" min="1" max="128">
        </div>
        <div class="form-group">
          <label>设备</label>
          <select id="device">
            <option value="0" selected>GPU (0)</option>
            <option value="cpu">CPU</option>
            <option value="0,1">GPU (0,1)</option>
          </select>
        </div>
      </div>
      <div class="form-group">
        <label style="display:flex;align-items:center;gap:8px;">
          <span id="llmStatus" style="font-size:0.8rem;padding:4px 10px;border-radius:10px;background:rgba(99,102,241,0.15);color:#a5b4fc;font-weight:500;">
            🔍 默认使用本地 LLM (qwen3.5) 逐帧视觉标注 — 直接理解语义返回坐标
          </span>
        </label>
      </div>
      <div class="form-group" style="margin-top:4px;">
        <label style="display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;padding:10px 12px;background:rgba(102,126,234,0.08);border-radius:8px;border:1px solid rgba(102,126,234,0.2);">
          <input type="checkbox" id="enableManualLabel" checked style="width:18px;height:18px;accent-color:#667eea;cursor:pointer;">
          <div>
            <div style="color:#e0e0e0;font-weight:500;">✏️ 启用人工补标（推荐）</div>
            <div style="color:#888;font-size:0.8rem;margin-top:2px;">LLM 标注完成后，未识别的图片将展示给您，可手动框选后再进入训练</div>
          </div>
        </label>
      </div>
    </div>

    <div class="btn-group">
      <button class="btn btn-primary" id="startBtn" onclick="startPipeline()">🚀 开始训练</button>
      <button class="btn btn-secondary" id="resetBtn" onclick="resetPipeline()">🔄 重置</button>
    </div>
  </div>

  <div class="card progress-section" id="progressSection">
    <h2>
      3. 训练进度
      <span class="status-badge status-idle" id="statusBadge">idle</span>
    </h2>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="progress-info">
      <span id="progressStep">等待开始...</span>
      <span id="progressPct">0%</span>
    </div>
    <div class="step-indicator" id="stepIndicator">
      <div class="step" data-step="extracting">📸 抽帧</div>
      <div class="step" data-step="labeling">🏷️ 标注</div>
      <div class="step" data-step="waiting_manual_label">🖍️ 人工补标</div>
      <div class="step" data-step="splitting">📊 分割</div>
      <div class="step" data-step="training">🧠 训练</div>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>

  <div class="card manual-section" id="manualSection">
    <h2>
      4. 🖍️ 人工补标
      <span class="status-badge status-running" style="margin-left:8px;">waiting_manual_label</span>
      <span class="manual-mode-label" id="manualModeLabel" style="float:right;font-weight:normal;font-size:0.85rem;color:#667eea;background:rgba(102,126,234,0.08);padding:2px 10px;border-radius:999px;">模式: -</span>
    </h2>

    <!-- ============= REVIEW_PROMPT: LLM 全标成功时的双按钮面板 ============= -->
    <div class="review-prompt-panel" id="reviewPromptPanel" style="display:none; background:linear-gradient(135deg, rgba(76,175,80,0.08), rgba(102,126,234,0.08)); border:1px dashed rgba(76,175,80,0.4); border-radius:14px; padding:28px; margin:12px 0 8px; text-align:center;">
      <div style="font-size:2.6rem; margin-bottom:10px;">🎉</div>
      <h3 style="margin:4px 0 14px; color:#2e7d32; font-size:1.3rem;">LLM 自动标注全部成功！</h3>
      <div class="rp-stats" style="display:flex; gap:16px; justify-content:center; margin:18px 0 24px; flex-wrap:wrap;">
        <div class="rp-stat"><div class="rp-num" id="rp_all">0</div><div class="rp-label">总帧数</div></div>
        <div class="rp-stat" style="color:#4caf50;"><div class="rp-num" id="rp_labeled">0</div><div class="rp-label">✅ 已成功标注</div></div>
        <div class="rp-stat" style="color:#ef5350;"><div class="rp-num" id="rp_unlabeled">0</div><div class="rp-label">❌ 待补标</div></div>
      </div>
      <div style="display:flex; gap:14px; justify-content:center; flex-wrap:wrap;">
        <button class="btn btn-primary" style="background:linear-gradient(90deg,#4caf50,#2e7d32); min-width:220px; font-size:1rem; padding:12px 22px;"
          onclick="manualCompleteAll()">
          ⏭️ 直接进入训练（跳过人工）
          <div style="font-size:0.75rem; opacity:0.85; font-weight:normal;">使用 LLM 标注的 100% 结果立刻开始训练</div>
        </button>
        <button class="btn btn-primary" style="background:linear-gradient(90deg,#667eea,#764ba2); min-width:220px; font-size:1rem; padding:12px 22px;"
          onclick="manualStartReviewAll()">
          🖍️ 开始复核全部帧
          <div style="font-size:0.75rem; opacity:0.85; font-weight:normal;">逐张检查 LLM 框选，可删除 / 调整 / 新增目标框</div>
        </button>
      </div>
      <div style="margin-top:20px; color:#777; font-size:0.82rem;">
        💡 <b>建议：</b>第一次训练 / 换目标类别时推荐先复核 5~10 张确认框选质量；
        同一类别再次训练且你已确认 LLM 表现稳定，可以直接点「进入训练」节省时间。
      </div>
    </div>

    <!-- ============= EDITOR (patch / review mode): Toolbar + Canvas + Actions ============= -->
    <div id="manualToolbar" style="display:none; margin-bottom:8px;">
      <button class="btn btn-secondary" id="switchReviewBtn" onclick="manualSwitchToReviewFromPatch()"
        style="background:linear-gradient(90deg,#ff9800,#f57c00); color:white; border:none; display:none;">
        🔁 切换到「复核全部帧」模式（查看 / 编辑 LLM 已标结果）
      </button>
    </div>

    <div class="manual-header" id="manualHeaderWrap" style="display:none;">
      <div class="manual-title" id="manualFrameTitle">frame_0001.jpg</div>
      <div class="manual-counter" id="manualCounter">0 / 0</div>
    </div>
    <div class="canvas-wrap" id="manualCanvasCage" style="display:none;">
      <canvas id="labelCanvas" width="800" height="600"></canvas>
    </div>
    <div id="manualActions" style="display:none;">
      <div class="manual-hint">
        💡 <b>操作说明：</b>
        <code>鼠标左键按住拖动</code> 绘制目标框 ｜
        <code>点击框右侧 ×</code> 或 <code>按 Delete/Backspace 键</code> 删除最后一个框 ｜
        <code>鼠标右键</code> 撤销上一个框 ｜
        所有框都需完整包住目标物体（保存时自动转为 YOLO 归一化坐标 class_id cx cy bw bh）。
      </div>
      <div class="box-list" id="boxList">
        <span class="empty-boxes">（暂无框，拖动鼠标在图上绘制第一个目标框）</span>
      </div>
      <div class="btn-group">
        <button class="btn btn-secondary" id="prevFrameBtn" onclick="manualPrevFrame()">⬅️ 上一张</button>
        <button class="btn btn-secondary" onclick="manualUndoLast()">↶ 撤销最后一个框</button>
        <button class="btn btn-secondary" onclick="manualClearAll()">🗑️ 清空所有</button>
        <button class="btn btn-secondary" id="skipFrameBtn" onclick="manualSkipFrame()">⏭️ 跳过此帧（背景/无目标）</button>
        <button class="btn btn-primary" id="saveFrameBtn" onclick="manualSaveFrame()">✅ 保存此帧并下一张</button>
      </div>
      <div class="btn-group" style="margin-top:12px;">
        <button class="btn btn-primary" style="background:linear-gradient(90deg,#4caf50,#2e7d32);" onclick="manualCompleteAll()">🎉 全部完成，开始训练</button>
        <span style="color:#888;font-size:0.82rem;align-self:center;margin-left:4px;">
          若不想处理剩余未标注帧，可直接点此进入训练（剩余帧将保持未标注状态）。
        </span>
      </div>
    </div>
  </div>

  <div class="card result-section" id="resultSection">
    <div class="result-box">
      <div class="icon">🎉</div>
      <h3>训练完成！</h3>
      <p id="resultInfo"></p>
      <a class="download-btn" id="downloadBtn" href="#">⬇️ 下载模型 (.pt)</a>
    </div>
  </div>
</div>

<script>
let videoPath = null;
let statusPoller = null;

const uploadArea = document.getElementById('uploadArea');
const videoFile = document.getElementById('videoFile');
const fileName = document.getElementById('fileName');

uploadArea.addEventListener('click', () => videoFile.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    videoFile.files = e.dataTransfer.files;
    handleFile();
  }
});
videoFile.addEventListener('change', handleFile);

checkLLMStatus();

async function checkLLMStatus() {
  try {
    const resp = await fetch('/api/llm_status');
    const data = await resp.json();
    const badge = document.getElementById('llmStatus');
    if (data.available) {
      badge.textContent = '🔍 大模型已连接 (qwen3.5) — 逐帧视觉标注，直接理解语义返回坐标';
      badge.style.background = 'rgba(76,175,80,0.2)';
      badge.style.color = '#81c784';
    } else {
      badge.textContent = '⚠️ 大模型未连接 — 请先启动 http://127.0.0.1:12345';
      badge.style.background = 'rgba(244,67,54,0.15)';
      badge.style.color = '#ef5350';
    }
  } catch(e) {
    const badge = document.getElementById('llmStatus');
    badge.textContent = '⚠️ 大模型未连接 — 请先启动 http://127.0.0.1:12345';
    badge.style.background = 'rgba(244,67,54,0.15)';
    badge.style.color = '#ef5350';
  }
}

function handleFile() {
  const f = videoFile.files[0];
  if (f) {
    fileName.textContent = `📎 ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`;
    uploadArea.style.borderColor = '#4caf50';
  }
}

function toggleSettings() {
  document.getElementById('settingsContent').classList.toggle('open');
}

function setStatusBadge(status) {
  const badge = document.getElementById('statusBadge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status;
}

function updateStepIndicator(activeStep) {
  document.querySelectorAll('.step').forEach(el => {
    const step = el.dataset.step;
    el.classList.remove('active', 'done');
    const stepOrder = ['extracting', 'init_model', 'labeling', 'waiting_manual_label', 'splitting', 'training', 'exporting'];
    const activeIdx = stepOrder.indexOf(activeStep);
    const elIdx = stepOrder.indexOf(step);
    if (activeIdx >= 0 && elIdx < activeIdx) el.classList.add('done');
    else if (step === activeStep) el.classList.add('active');
  });
}

async function startPipeline() {
  if (!videoFile.files.length) {
    alert('请先上传视频文件');
    return;
  }
  const className = document.getElementById('className').value.trim();
  if (!className) {
    alert('请填写识别目标名称');
    return;
  }

  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = '⏳ 处理中...';

  const formData = new FormData();
  formData.append('video', videoFile.files[0]);

  try {
    const uploadResp = await fetch('/api/upload', { method: 'POST', body: formData });
    const uploadData = await uploadResp.json();
    if (uploadData.error) throw new Error(uploadData.error);
    videoPath = uploadData.video_path;

    document.getElementById('progressSection').classList.add('active');
    document.getElementById('logArea').innerHTML = '';

    const processResp = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_path: videoPath,
        class_name: className,
        fps: parseFloat(document.getElementById('fps').value),
        epochs: parseInt(document.getElementById('epochs').value),
        imgsz: parseInt(document.getElementById('imgsz').value),
        batch: parseInt(document.getElementById('batch').value),
        device: document.getElementById('device').value,
        workers: 0,
        train_ratio: parseFloat(document.getElementById('trainRatio').value),
        enable_manual_label: document.getElementById('enableManualLabel').checked,
      })
    });

    if (processResp.status === 200) {
      startPolling();
    } else {
      const err = await processResp.json();
      throw new Error(err.error || 'Failed to start');
    }
  } catch (e) {
    alert('错误: ' + e.message);
    resetPipeline();
  }
}

function startPolling() {
  if (statusPoller) clearInterval(statusPoller);
  statusPoller = setInterval(async () => {
    try {
      const resp = await fetch('/api/status');
      const data = await resp.json();
      updateUI(data);
      if (data.status === 'done' || data.status === 'error') {
        clearInterval(statusPoller);
        statusPoller = null;
      }
    } catch (e) {
      console.error('Status poll error:', e);
    }
  }, 500);
}

function updateUI(data) {
  document.getElementById('progressFill').style.width = data.progress + '%';
  document.getElementById('progressPct').textContent = data.progress + '%';
  document.getElementById('progressStep').textContent = data.message || data.step;
  setStatusBadge(data.status);
  updateStepIndicator(data.step);

  const logArea = document.getElementById('logArea');
  if (data.logs && data.logs.length) {
    logArea.innerHTML = data.logs.map(l => {
      const cls = data.status === 'error' ? 'error' : '';
      return `<div class="log-line ${cls}">${l}</div>`;
    }).join('');
    logArea.scrollTop = logArea.scrollHeight;
  }

  if (data.status === 'done' && data.result) {
    document.getElementById('resultSection').classList.add('active');
    document.getElementById('resultInfo').innerHTML =
      `类别: <b>${data.result.class_name}</b> | 抽帧: ${data.result.frames} | 标注: ${data.result.labeled}`;
    document.getElementById('downloadBtn').href = '/api/download/' + encodeURIComponent(data.result.model_name);
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').textContent = '🚀 重新训练';
  }

  if (data.status === 'error') {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').textContent = '🚀 开始训练';
  }
}

async function resetPipeline() {
  if (statusPoller) { clearInterval(statusPoller); statusPoller = null; }
  try { await fetch('/api/reset', { method: 'POST' }); } catch(e) {}
  resetUI();
}

function resetUI() {
  videoPath = null;
  videoFile.value = '';
  fileName.textContent = '';
  uploadArea.style.borderColor = '';
  document.getElementById('progressSection').classList.remove('active');
  document.getElementById('resultSection').classList.remove('active');
  document.getElementById('manualSection').classList.remove('active');
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressPct').textContent = '0%';
  document.getElementById('progressStep').textContent = '等待开始...';
  document.getElementById('logArea').innerHTML = '';
  setStatusBadge('idle');
  document.querySelectorAll('.step').forEach(el => el.classList.remove('active', 'done'));
  document.getElementById('startBtn').disabled = false;
  document.getElementById('startBtn').textContent = '🚀 开始训练';
  manualTeardown();
}

/* ==================== Manual Labeling (Canvas-based) ==================== */
const _MANUAL = {
  runId: null,
  mode: 'patch',       // 'patch' | 'review_prompt' | 'review'
  todo: [],            // full list of basenames from /api/manual_label/todo
  preSavedBoxes: {},   // filename -> [{x1,y1,x2,y2}] existing LLM bboxes (for review mode)
  stats: { all: 0, labeled: 0, unlabeled: 0 },
  curIdx: -1,          // current index within todo
  boxes: [],           // boxes on current frame: [{x1,y1,x2,y2}] all 0..1 normalized vs NATURAL image size
  natW: 0,             // natural (original) image width for current frame
  natH: 0,             // natural (original) image height for current frame
  dispW: 0,            // canvas display width (CSS pixels) for coord mapping
  dispH: 0,
  drawing: false,
  drawStart: null,     // {x,y} in 0..1 normalized coords
  drawCur: null,
  _img: null,
  _entered: false,
  _pollingWasOn: false,
};

function manualTeardown() {
  const c = document.getElementById('labelCanvas');
  if (c) {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  }
  Object.assign(_MANUAL, {
    runId: null, mode: 'patch', todo: [], preSavedBoxes: {},
    stats: { all: 0, labeled: 0, unlabeled: 0 },
    curIdx: -1, boxes: [],
    natW: 0, natH: 0, dispW: 0, dispH: 0,
    drawing: false, drawStart: null, drawCur: null,
    _img: null, _entered: false,
  });
  const panel = document.getElementById('reviewPromptPanel');
  if (panel) panel.style.display = 'none';
  const toolbar = document.getElementById('manualToolbar');
  if (toolbar) toolbar.style.display = '';
  const header = document.getElementById('manualHeaderWrap');
  if (header) header.style.display = '';
  const cage = document.getElementById('manualCanvasCage');
  if (cage) cage.style.display = '';
  const actions = document.getElementById('manualActions');
  if (actions) actions.style.display = '';
}

async function manualEnterIfNeeded(data) {
  if (_MANUAL._entered) return;
  if (data.step !== 'waiting_manual_label') return;
  _MANUAL._entered = true;
  try {
    const resp = await fetch('/api/manual_label/todo');
    const todoData = await resp.json();
    if (!todoData.run_id || !todoData.todo || todoData.todo.length === 0) {
      await manualCompleteAll(true);
      return;
    }
    _MANUAL.runId = todoData.run_id;
    _MANUAL.mode = todoData.mode || 'patch';
    _MANUAL.preSavedBoxes = todoData.pre_saved_boxes || {};
    _MANUAL.stats = {
      all: todoData.all_count || todoData.todo.length,
      labeled: todoData.labeled_count || 0,
      unlabeled: (todoData.unlabeled_count != null) ? todoData.unlabeled_count : 0,
    };
    document.getElementById('manualSection').classList.add('active');
    _bindCanvasEvents();

    if (_MANUAL.mode === 'review_prompt') {
      // --- ALL frames LLM-labeled successfully. Show two big buttons. ---
      manualShowReviewPrompt();
    } else {
      // --- Regular mode (patch or already in review): show canvas editor. ---
      const doneSet = new Set(todoData.done || []);
      const remaining = (todoData.todo || []).filter(f => !doneSet.has(f));
      _MANUAL.todo = remaining.length ? remaining : (todoData.todo || []);
      _MANUAL.curIdx = 0;
      manualShowEditorUI();
      await manualLoadFrame(0);
    }
  } catch (e) {
    console.error('manualEnter failed:', e);
    alert('进入人工补标失败: ' + e.message);
  }
}

function manualShowEditorUI() {
  const prompt = document.getElementById('reviewPromptPanel');
  if (prompt) prompt.style.display = 'none';
  const tb = document.getElementById('manualToolbar');
  if (tb) tb.style.display = '';
  const header = document.getElementById('manualHeaderWrap');
  if (header) header.style.display = '';
  const cage = document.getElementById('manualCanvasCage');
  if (cage) cage.style.display = '';
  const acts = document.getElementById('manualActions');
  if (acts) acts.style.display = '';
  // Update mode label at top-right
  const ml = document.getElementById('manualModeLabel');
  if (ml) {
    if (_MANUAL.mode === 'review') ml.textContent = '模式: 复核全部（编辑 LLM 已有框）';
    else ml.textContent = '模式: 补标未成功帧（空白 canvas）';
  }
  // Always enable "switch to review all" when in patch mode
  const sw = document.getElementById('switchReviewBtn');
  if (sw) sw.style.display = (_MANUAL.mode === 'patch') ? '' : 'none';
}

function manualShowReviewPrompt() {
  const prompt = document.getElementById('reviewPromptPanel');
  if (!prompt) return;
  prompt.style.display = '';
  const tb = document.getElementById('manualToolbar');
  if (tb) tb.style.display = 'none';
  const header = document.getElementById('manualHeaderWrap');
  if (header) header.style.display = 'none';
  const cage = document.getElementById('manualCanvasCage');
  if (cage) cage.style.display = 'none';
  const acts = document.getElementById('manualActions');
  if (acts) acts.style.display = 'none';
  const ml = document.getElementById('manualModeLabel');
  if (ml) ml.textContent = '模式: 选择操作（LLM 已全标成功）';
  // Fill stats
  document.getElementById('rp_all').textContent = _MANUAL.stats.all;
  document.getElementById('rp_labeled').textContent = _MANUAL.stats.labeled;
  document.getElementById('rp_unlabeled').textContent = _MANUAL.stats.unlabeled;
}

async function manualStartReviewAll() {
  try {
    const resp = await fetch('/api/manual_label/review_all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const d = await resp.json();
    if (!resp.ok || !d.ok) throw new Error(d.error || '切换到复核模式失败');
    // Re-fetch todo for fresh todo list + pre_saved_boxes for all frames
    const t = await fetch('/api/manual_label/todo');
    const todoData = await t.json();
    _MANUAL.mode = todoData.mode || 'review';
    _MANUAL.preSavedBoxes = todoData.pre_saved_boxes || {};
    _MANUAL.todo = (todoData.todo || []).slice();
    _MANUAL.curIdx = 0;
    manualShowEditorUI();
    await manualLoadFrame(0);
  } catch (e) {
    alert('切换到复核模式失败: ' + e.message);
  }
}

async function manualSwitchToReviewFromPatch() {
  // Same backend call; convenience function so the toolbar button onclick can differ
  await manualStartReviewAll();
}

function _bindCanvasEvents() {
  const c = document.getElementById('labelCanvas');
  if (c._bound) return;
  c._bound = true;
  c.addEventListener('mousedown', onCanvasMouseDown);
  c.addEventListener('mousemove', onCanvasMouseMove);
  c.addEventListener('mouseup', onCanvasMouseUp);
  c.addEventListener('mouseleave', onCanvasMouseUp);
  c.addEventListener('contextmenu', e => {
    e.preventDefault();
    manualUndoLast();
  });
  document.addEventListener('keydown', e => {
    if (!_MANUAL._entered || !document.getElementById('manualSection').classList.contains('active')) return;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      // Don't trigger while focus is inside an input; label page has no inputs but guard anyway
      const tag = (document.activeElement && document.activeElement.tagName) || '';
      if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        manualUndoLast();
      }
    }
  });
}

function _canvasToNorm(c, clientX, clientY) {
  // Map client (page) coords → canvas element coords → 0..1 normalized vs displayed portion
  // (letterboxed inside canvas). We use the actual CSS size for hit testing, then translate
  // to natural 0..1 via the letterbox offset we recorded at draw time.
  const rect = c.getBoundingClientRect();
  const cx = clientX - rect.left;
  const cy = clientY - rect.top;
  const cw = rect.width;
  const ch = rect.height;
  // Letterbox: image fits inside cw×ch preserving aspect, centered.
  const iw = _MANUAL.natW || cw;
  const ih = _MANUAL.natH || ch;
  const scale = Math.min(cw / iw, ch / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  const offX = (cw - dw) / 2;
  const offY = (ch - dh) / 2;
  _MANUAL.dispW = cw;
  _MANUAL.dispH = ch;
  // Clamp cx,cy inside the drawn image rectangle (ignore clicks on padding bars)
  const ix = Math.max(0, Math.min(dw, cx - offX));
  const iy = Math.max(0, Math.min(dh, cy - offY));
  // Convert to 0..1 vs natural image dimensions (this is what we store in boxes[])
  const nx = dw > 0 ? ix / dw : 0;
  const ny = dh > 0 ? iy / dh : 0;
  return { nx, ny };
}

function onCanvasMouseDown(e) {
  if (e.button !== 0) return;
  const c = document.getElementById('labelCanvas');
  const { nx, ny } = _canvasToNorm(c, e.clientX, e.clientY);
  _MANUAL.drawing = true;
  _MANUAL.drawStart = { x: nx, y: ny };
  _MANUAL.drawCur = { x: nx, y: ny };
  manualRedraw();
}
function onCanvasMouseMove(e) {
  if (!_MANUAL.drawing) return;
  const c = document.getElementById('labelCanvas');
  const { nx, ny } = _canvasToNorm(c, e.clientX, e.clientY);
  _MANUAL.drawCur = { x: nx, y: ny };
  manualRedraw();
}
function onCanvasMouseUp(e) {
  if (!_MANUAL.drawing) return;
  const c = document.getElementById('labelCanvas');
  let endP;
  if (e && e.clientX != null) {
    endP = _canvasToNorm(c, e.clientX, e.clientY);
  } else {
    endP = _MANUAL.drawCur || _MANUAL.drawStart;
  }
  _MANUAL.drawing = false;
  const s = _MANUAL.drawStart;
  const x1 = Math.min(s.x, endP.nx);
  const y1 = Math.min(s.y, endP.ny);
  const x2 = Math.max(s.x, endP.nx);
  const y2 = Math.max(s.y, endP.ny);
  _MANUAL.drawStart = null;
  _MANUAL.drawCur = null;
  // Ignore tiny boxes (accidental click)
  if ((x2 - x1) < 0.005 || (y2 - y1) < 0.005) {
    manualRedraw();
    return;
  }
  _MANUAL.boxes.push({ x1, y1, x2, y2 });
  manualRedraw();
  renderBoxList();
}

function manualRedraw() {
  const c = document.getElementById('labelCanvas');
  if (!_MANUAL._img) return;
  const ctx = c.getContext('2d');
  // Resize canvas backing store to CSS-rendered size × devicePixelRatio for crispness
  const rect = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(10, Math.round(rect.width));
  const cssH = Math.max(10, Math.round(rect.height));
  // Only resize when needed to avoid flickering
  if (c.width !== Math.round(cssW * dpr) || c.height !== Math.round(cssH * dpr)) {
    c.width = Math.round(cssW * dpr);
    c.height = Math.round(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  // Draw image letterboxed into cssW×cssH
  const iw = _MANUAL.natW || _MANUAL._img.naturalWidth || cssW;
  const ih = _MANUAL.natH || _MANUAL._img.naturalHeight || cssH;
  const scale = Math.min(cssW / iw, cssH / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  const offX = (cssW - dw) / 2;
  const offY = (cssH - dh) / 2;
  ctx.drawImage(_MANUAL._img, offX, offY, dw, dh);
  // Helper: convert 0..1 box (natural image) to canvas CSS coords
  const toCss = b => ({
    x: offX + b.x1 * dw,
    y: offY + b.y1 * dh,
    w: (b.x2 - b.x1) * dw,
    h: (b.y2 - b.y1) * dh,
  });
  // Draw committed boxes
  _MANUAL.boxes.forEach((b, i) => {
    const r = toCss(b);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#667eea';
    ctx.fillStyle = 'rgba(102,126,234,0.15)';
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    // Label chip
    const label = `#${i + 1}`;
    ctx.font = '12px "Segoe UI", sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = '#667eea';
    ctx.fillRect(r.x, Math.max(0, r.y - 16), tw + 8, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, r.x + 4, Math.max(12, r.y - 4));
  });
  // Draw in-progress drag rectangle
  if (_MANUAL.drawing && _MANUAL.drawStart && _MANUAL.drawCur) {
    const s = _MANUAL.drawStart, e = _MANUAL.drawCur;
    const b = {
      x1: Math.min(s.x, e.x), y1: Math.min(s.y, e.y),
      x2: Math.max(s.x, e.x), y2: Math.max(s.y, e.y),
    };
    const r = toCss(b);
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = '#fbbf24';
    ctx.fillStyle = 'rgba(251,191,36,0.12)';
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]);
  }
}

function renderBoxList() {
  const list = document.getElementById('boxList');
  if (!_MANUAL.boxes.length) {
    list.innerHTML = '<span class="empty-boxes">（暂无框，拖动鼠标在图上绘制第一个目标框）</span>';
    return;
  }
  list.innerHTML = _MANUAL.boxes.map((b, i) => {
    const cx = ((b.x1 + b.x2) / 2).toFixed(3);
    const cy = ((b.y1 + b.y2) / 2).toFixed(3);
    const bw = (b.x2 - b.x1).toFixed(3);
    const bh = (b.y2 - b.y1).toFixed(3);
    return `<span class="box-chip">
      框#${i + 1} cx=${cx} cy=${cy} bw=${bw} bh=${bh}
      <span class="x-btn" onclick="manualDeleteBox(${i})" title="删除此框">×</span>
    </span>`;
  }).join('');
}

function manualDeleteBox(i) {
  if (!_MANUAL.boxes[i]) return;
  _MANUAL.boxes.splice(i, 1);
  manualRedraw();
  renderBoxList();
}
function manualUndoLast() {
  if (_MANUAL.boxes.length) {
    _MANUAL.boxes.pop();
    manualRedraw();
    renderBoxList();
  }
}
function manualClearAll() {
  _MANUAL.boxes = [];
  manualRedraw();
  renderBoxList();
}

async function manualLoadFrame(idx) {
  if (!_MANUAL.todo.length) return;
  if (idx < 0) idx = 0;
  if (idx >= _MANUAL.todo.length) idx = _MANUAL.todo.length - 1;
  _MANUAL.curIdx = idx;
  const filename = _MANUAL.todo[idx];
  document.getElementById('manualFrameTitle').textContent = filename;
  const modeTxt = _MANUAL.mode === 'review'
    ? `复核全部帧：${idx + 1} / ${_MANUAL.todo.length}  （共 ${_MANUAL.stats.all} 张，LLM 已标 ${_MANUAL.stats.labeled}，未标 ${_MANUAL.stats.unlabeled}）`
    : `补标：${idx + 1} / ${_MANUAL.todo.length}  （总 ${_MANUAL.stats.all} / 待补标 ${_MANUAL.todo.length}）`;
  document.getElementById('manualCounter').textContent = modeTxt;
  const url = `/api/frame/${_MANUAL.runId}/${encodeURIComponent(filename)}`;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    _MANUAL._img = img;
    _MANUAL.natW = img.naturalWidth;
    _MANUAL.natH = img.naturalHeight;
    // --- REVIEW MODE: load pre-saved LLM boxes onto canvas so user can edit/delete/add ---
    const saved = (_MANUAL.preSavedBoxes && _MANUAL.preSavedBoxes[filename]) || [];
    if (saved.length) {
      _MANUAL.boxes = saved.map(b => ({
        x1: Number(b.x1) || 0, y1: Number(b.y1) || 0,
        x2: Number(b.x2) || 0, y2: Number(b.y2) || 0,
      })).filter(b => b.x2 - b.x1 > 0.0005 && b.y2 - b.y1 > 0.0005);
    } else {
      _MANUAL.boxes = [];
    }
    renderBoxList();
    requestAnimationFrame(() => {
      manualRedraw();
      window.addEventListener('resize', _onWinResize, { once: true });
    });
  };
  img.onerror = () => {
    alert(`加载图片失败: ${filename}`);
  };
  img.src = url;
  document.getElementById('prevFrameBtn').disabled = idx <= 0;
}
let __resizeRaf = 0;
function _onWinResize() {
  if (!__resizeRaf) {
    __resizeRaf = requestAnimationFrame(() => {
      __resizeRaf = 0;
      if (_MANUAL._img) manualRedraw();
    });
  }
  window.addEventListener('resize', _onWinResize, { once: true });
}

function manualPrevFrame() {
  if (_MANUAL.curIdx > 0) manualLoadFrame(_MANUAL.curIdx - 1);
}

async function manualSaveFrame() {
  const filename = _MANUAL.todo[_MANUAL.curIdx];
  if (!filename) return;
  // Convert boxes[x1,y1,x2,y2] 0..1 → YOLO [cx,cy,bw,bh] 0..1
  const yolo = _MANUAL.boxes.map(b => {
    const cx = (b.x1 + b.x2) / 2;
    const cy = (b.y1 + b.y2) / 2;
    const bw = b.x2 - b.x1;
    const bh = b.y2 - b.y1;
    return [cx, cy, bw, bh];
  });
  try {
    const resp = await fetch('/api/manual_label/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, boxes: yolo }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || 'save failed');
    // Advance to next frame, or if last one, prompt complete
    if (_MANUAL.curIdx + 1 < _MANUAL.todo.length) {
      await manualLoadFrame(_MANUAL.curIdx + 1);
    } else {
      // Last frame saved
      const goOn = confirm(
        `✅ 已保存最后一张 ${filename}（${yolo.length} 个框）。\n要结束人工补标并开始训练吗？\n点"取消"可留在本页继续修改。`
      );
      if (goOn) await manualCompleteAll(true);
    }
  } catch (e) {
    alert('保存失败: ' + e.message);
  }
}

async function manualSkipFrame() {
  const filename = _MANUAL.todo[_MANUAL.curIdx];
  if (!filename) return;
  const ok = confirm(`⏭️ 将 "${filename}" 标记为"无目标/背景帧"（写入空 .txt，作为 YOLO 负样本参与训练）。\n确定跳过？`);
  if (!ok) return;
  try {
    const resp = await fetch('/api/manual_label/skip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || 'skip failed');
    if (_MANUAL.curIdx + 1 < _MANUAL.todo.length) {
      await manualLoadFrame(_MANUAL.curIdx + 1);
    } else {
      const goOn = confirm(`⏭️ 已跳过最后一张。要结束人工补标并开始训练吗？`);
      if (goOn) await manualCompleteAll(true);
    }
  } catch (e) {
    alert('跳过失败: ' + e.message);
  }
}

async function manualCompleteAll(silent) {
  if (!silent) {
    const todo = _MANUAL.todo.length;
    let processed = 0;
    try {
      const resp = await fetch('/api/manual_label/todo');
      const d = await resp.json();
      processed = (d.done || []).length;
    } catch (_) { /* ignore */ }
    const left = Math.max(0, todo - processed);
    const msg = left > 0
      ? `还有 ${left} / ${todo} 张未处理（保持原未标注状态，不会进入训练集）。\n确定结束人工补标，开始训练？`
      : `已处理 ${processed} / ${todo} 张。确定结束人工补标，开始训练？`;
    if (!confirm(msg)) return;
  }
  try {
    const resp = await fetch('/api/manual_label/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || 'complete failed');
  } catch (e) {
    console.warn('manualComplete signal error:', e);
  }
  // Hide manual section, keep polling (progress/training next)
  document.getElementById('manualSection').classList.remove('active');
  _MANUAL._entered = false;
}

/* Patch updateUI: call manualEnterIfNeeded when step=waiting_manual_label */
(function patchUpdateUI() {
  // Preserve the original updateUI defined above and wrap it.
  const _origUpdateUI = window.updateUI;
  window.updateUI = function (data) {
    _origUpdateUI(data);
    if (data.step === 'waiting_manual_label' && data.status !== 'error' && data.status !== 'done') {
      manualEnterIfNeeded(data);
    } else if (data.status === 'done' || data.status === 'error') {
      // Make sure manual section hides on run end
      document.getElementById('manualSection').classList.remove('active');
      _MANUAL._entered = false;
    }
  };
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO 视频训练平台 启动中...")
    print("  访问地址: http://127.0.0.1:5000")
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
