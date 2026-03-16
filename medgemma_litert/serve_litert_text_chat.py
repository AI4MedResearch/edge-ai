#!/usr/bin/env python3
"""Simple HTTP chat endpoint for LiteRT text inference."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tokenizers import Tokenizer


def _shape_tuple(detail: dict[str, Any]) -> tuple[int, ...]:
  return tuple(int(x) for x in detail["shape"])


class LiteRTTextChatEngine:
  """Runs text-only chat inference over prefill/decode signatures."""

  def __init__(self, model_path: str, embedder_path: str, tokenizer_path: str):
    self._lock = threading.Lock()
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.eos_id = self.tokenizer.token_to_id("<eos>")
    self.eot_id = self.tokenizer.token_to_id("<end_of_turn>")

    self.embedder = Interpreter(embedder_path)
    self.embedder.allocate_tensors()
    self.embed_decode = self.embedder.get_signature_runner("decode_embedder")

    self.model = Interpreter(model_path)
    self.model.allocate_tensors()

    signatures = self.model.get_signature_list()
    prefill_lens = []
    for name in signatures:
      if name.startswith("prefill_"):
        try:
          prefill_lens.append(int(name.split("_", 1)[1]))
        except ValueError:
          continue
    if not prefill_lens:
      raise RuntimeError("No prefill_* signatures found in model.")
    self.prefill_len = max(prefill_lens)
    self.prefill = self.model.get_signature_runner(f"prefill_{self.prefill_len}")
    self.decode = self.model.get_signature_runner("decode")

    embed_sigs = self.embedder.get_signature_list()
    self.embed_prefill = None
    embed_prefill_name = f"prefill_embedder_{self.prefill_len}"
    if embed_prefill_name in embed_sigs:
      self.embed_prefill = self.embedder.get_signature_runner(embed_prefill_name)

    self.prefill_inputs = self.prefill.get_input_details()
    self.decode_inputs = self.decode.get_input_details()
    self.cache_len = _shape_tuple(self.decode_inputs["mask"])[-1]
    self.embed_dim = _shape_tuple(self.decode_inputs["embeddings"])[-1]

  def _content_to_text(self, content: Any) -> str:
    if isinstance(content, str):
      return content.strip()
    if isinstance(content, list):
      out = []
      for item in content:
        if not isinstance(item, dict):
          continue
        if item.get("type") == "text":
          out.append(str(item.get("text", "")))
      return "".join(out).strip()
    return str(content).strip()

  def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
    if not messages:
      raise ValueError("messages must not be empty")

    first_user_prefix = ""
    idx = 0
    if messages[0].get("role") == "system":
      first_user_prefix = self._content_to_text(messages[0].get("content", "")) + "\n\n"
      idx = 1

    out = ["<bos>"]
    loop_messages = messages[idx:]
    for i, message in enumerate(loop_messages):
      role = message.get("role")
      expected = "user" if i % 2 == 0 else "assistant"
      if role != expected:
        raise ValueError(
            f"Conversation roles must alternate user/assistant. got={role}, expected={expected}"
        )
      role_tag = "model" if role == "assistant" else role
      content = self._content_to_text(message.get("content", ""))
      if i == 0 and first_user_prefix:
        content = first_user_prefix + content
      out.append(f"<start_of_turn>{role_tag}\n{content}<end_of_turn>\n")
    out.append("<start_of_turn>model\n")
    return "".join(out)

  def _make_prefill_mask(self, block_len: int, start_pos: int) -> np.ndarray:
    mask = np.full((1, 1, block_len, self.cache_len), -np.inf, dtype=np.float32)
    for i in range(block_len):
      hi = min(start_pos + i + 1, self.cache_len)
      mask[0, 0, i, :hi] = 0.0
    return mask

  def _make_decode_mask(self, pos: int) -> np.ndarray:
    mask = np.full((1, 1, 1, self.cache_len), -np.inf, dtype=np.float32)
    hi = min(pos + 1, self.cache_len)
    mask[0, 0, 0, :hi] = 0.0
    return mask

  def _pick_logits(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
    for key, value in outputs.items():
      if "logit" in key.lower():
        return value
    return next(iter(outputs.values()))

  def _extract_kv(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v for k, v in outputs.items() if k.startswith("kv_cache_")}

  def _build_embeddings(self, token_ids: list[int]) -> np.ndarray:
    rows = []
    pos = 0
    if self.embed_prefill is not None:
      while len(token_ids) - pos >= self.prefill_len:
        block = np.array([token_ids[pos : pos + self.prefill_len]], dtype=np.int32)
        embs = self.embed_prefill(token_ids=block)["embeddings"][0]
        rows.extend(embs.astype(np.float32))
        pos += self.prefill_len
    while pos < len(token_ids):
      emb = self.embed_decode(
          token_ids=np.array([[token_ids[pos]]], dtype=np.int32)
      )["embeddings"][0, 0]
      rows.append(emb.astype(np.float32))
      pos += 1
    return np.stack(rows, axis=0)

  def generate(self, messages: list[dict[str, Any]], max_new_tokens: int = 64) -> dict[str, Any]:
    with self._lock:
      t0 = time.time()
      prompt = self._build_prompt(messages)
      prompt_ids = self.tokenizer.encode(prompt).ids
      if not prompt_ids:
        raise ValueError("Prompt tokenization produced no ids.")

      context = self._build_embeddings(prompt_ids)
      if context.shape[0] >= self.cache_len:
        raise ValueError(
            f"Context too long ({context.shape[0]} tokens). cache_len={self.cache_len}."
        )

      kv = {}
      for name, detail in self.decode_inputs.items():
        if name.startswith("kv_cache_"):
          kv[name] = np.zeros(_shape_tuple(detail), dtype=detail["dtype"])

      # Consume all but final context token.
      pos = 0
      prefix = context[:-1]
      final_emb = context[-1]
      while prefix.shape[0] - pos >= self.prefill_len:
        block = prefix[pos : pos + self.prefill_len][None, :, :].astype(np.float32)
        feed = {}
        for name, detail in self.prefill_inputs.items():
          dtype = detail["dtype"]
          if name == "embeddings":
            feed[name] = block.astype(dtype)
          elif name == "input_pos":
            feed[name] = np.arange(pos, pos + self.prefill_len, dtype=np.int32).astype(dtype)
          elif name == "mask":
            feed[name] = self._make_prefill_mask(self.prefill_len, pos).astype(dtype)
          elif name.startswith("kv_cache_"):
            feed[name] = kv[name].astype(dtype)
          else:
            feed[name] = np.zeros(_shape_tuple(detail), dtype=dtype)
        out = self.prefill(**feed)
        kv = self._extract_kv(out)
        pos += self.prefill_len

      while pos < prefix.shape[0]:
        feed = {}
        for name, detail in self.decode_inputs.items():
          dtype = detail["dtype"]
          if name == "embeddings":
            feed[name] = prefix[pos][None, None, :].astype(dtype)
          elif name == "input_pos":
            feed[name] = np.array([pos], dtype=np.int32).astype(dtype)
          elif name == "mask":
            feed[name] = self._make_decode_mask(pos).astype(dtype)
          elif name.startswith("kv_cache_"):
            feed[name] = kv[name].astype(dtype)
          else:
            feed[name] = np.zeros(_shape_tuple(detail), dtype=dtype)
        out = self.decode(**feed)
        kv = self._extract_kv(out)
        pos += 1

      # Decode final context token to get first generation logits.
      final_pos = context.shape[0] - 1
      feed = {}
      for name, detail in self.decode_inputs.items():
        dtype = detail["dtype"]
        if name == "embeddings":
          feed[name] = final_emb[None, None, :].astype(dtype)
        elif name == "input_pos":
          feed[name] = np.array([final_pos], dtype=np.int32).astype(dtype)
        elif name == "mask":
          feed[name] = self._make_decode_mask(final_pos).astype(dtype)
        elif name.startswith("kv_cache_"):
          feed[name] = kv[name].astype(dtype)
        else:
          feed[name] = np.zeros(_shape_tuple(detail), dtype=dtype)
      out = self.decode(**feed)
      kv = self._extract_kv(out)
      logits = self._pick_logits(out).reshape(-1, self._pick_logits(out).shape[-1])[-1]

      generated_ids = []
      next_id = int(np.argmax(logits))
      for step in range(max_new_tokens):
        generated_ids.append(next_id)
        if (self.eos_id is not None and next_id == self.eos_id) or (
            self.eot_id is not None and next_id == self.eot_id
        ):
          break

        emb = self.embed_decode(
            token_ids=np.array([[next_id]], dtype=np.int32)
        )["embeddings"].astype(np.float32)
        token_pos = final_pos + step + 1
        feed = {}
        for name, detail in self.decode_inputs.items():
          dtype = detail["dtype"]
          if name == "embeddings":
            feed[name] = emb.astype(dtype)
          elif name == "input_pos":
            feed[name] = np.array([token_pos], dtype=np.int32).astype(dtype)
          elif name == "mask":
            feed[name] = self._make_decode_mask(token_pos).astype(dtype)
          elif name.startswith("kv_cache_"):
            feed[name] = kv[name].astype(dtype)
          else:
            feed[name] = np.zeros(_shape_tuple(detail), dtype=dtype)
        out = self.decode(**feed)
        kv = self._extract_kv(out)
        logits = self._pick_logits(out).reshape(-1, self._pick_logits(out).shape[-1])[-1]
        next_id = int(np.argmax(logits))

      raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
      clean_text = raw_text
      for marker in ("<end_of_turn>", "<eos>"):
        if marker in clean_text:
          clean_text = clean_text.split(marker, 1)[0]
      clean_text = clean_text.strip()

      t1 = time.time()
      return {
          "reply": clean_text,
          "raw_reply": raw_text,
          "generated_ids": generated_ids,
          "prompt_token_count": len(prompt_ids),
          "completion_token_count": len(generated_ids),
          "elapsed_s": round(t1 - t0, 3),
      }


class ChatHandler(BaseHTTPRequestHandler):
  engine: LiteRTTextChatEngine | None = None
  model_name = "medgemma-litert-text"

  def _send_json(self, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.end_headers()
    self.wfile.write(body)

  def do_OPTIONS(self) -> None:  # noqa: N802
    self._send_json(200, {"ok": True})

  def do_GET(self) -> None:  # noqa: N802
    if self.path == "/health":
      self._send_json(200, {"ok": True})
      return
    self._send_json(404, {"error": "not_found"})

  def do_POST(self) -> None:  # noqa: N802
    if self.engine is None:
      self._send_json(500, {"error": "engine_not_initialized"})
      return

    if self.path not in ("/chat", "/v1/chat/completions"):
      self._send_json(404, {"error": "not_found"})
      return

    try:
      content_len = int(self.headers.get("Content-Length", "0"))
      raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
      payload = json.loads(raw.decode("utf-8"))
      messages = payload.get("messages")
      if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
      max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 64)))
      max_tokens = max(1, min(max_tokens, 256))
      result = self.engine.generate(messages=messages, max_new_tokens=max_tokens)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._send_json(400, {"error": str(exc)})
      return

    if self.path == "/v1/chat/completions":
      now = int(time.time())
      response = {
          "id": f"chatcmpl-{now}",
          "object": "chat.completion",
          "created": now,
          "model": self.model_name,
          "choices": [
              {
                  "index": 0,
                  "message": {"role": "assistant", "content": result["reply"]},
                  "finish_reason": "stop",
              }
          ],
          "usage": {
              "prompt_tokens": result["prompt_token_count"],
              "completion_tokens": result["completion_token_count"],
              "total_tokens": result["prompt_token_count"] + result["completion_token_count"],
          },
          "debug": {
              "elapsed_s": result["elapsed_s"],
              "raw_reply": result["raw_reply"],
              "generated_ids": result["generated_ids"],
          },
      }
      self._send_json(200, response)
      return

    self._send_json(200, result)

  def log_message(self, fmt: str, *args: Any) -> None:
    # Keep logs concise; server prints startup details separately.
    print(f"[http] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Serve LiteRT text chat endpoint.")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--model", default="/tmp/medgemma_local/model.tflite")
  parser.add_argument("--embedder", default="/tmp/medgemma_local/embedder.tflite")
  parser.add_argument("--tokenizer", default="/tmp/medgemma_local/tokenizer.json")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  engine = LiteRTTextChatEngine(
      model_path=args.model, embedder_path=args.embedder, tokenizer_path=args.tokenizer
  )
  ChatHandler.engine = engine
  server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
  print(f"server_started host={args.host} port={args.port}")
  print(f"model={args.model}")
  print(f"embedder={args.embedder}")
  print(f"tokenizer={args.tokenizer}")
  server.serve_forever()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
