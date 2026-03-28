"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompts to use
- How to analyze each frame
- When to submit a guess vs. gather more context

Architecture:
  AI #1 (Extractor) — runs every frame, extracts context clues and decides
                       when there is enough evidence to guess.
  AI #2 (Guesser)   — fires once when the extractor says ready, produces the
                       final answer using all accumulated frames + clues.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from core import Frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FRAMES = 10  # Rolling window of frames kept in context

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = """\
You are a context extractor for a visual charade guessing game. You receive \
numbered sequential frames (Frame 1, Frame 2, ...) from a live camera feed. \
A human is acting out or miming a word or phrase (like in charades).

Your job:
1. Track how the person's actions CHANGE across frames — note movement progression, \
   gesture sequences, and transitions between poses.
2. For each new or evolving action, write a detailed description that captures: \
   which body part is moving, the direction/arc of the movement, the speed implied \
   by frame-to-frame change, and what real-world action it resembles.
3. Decide whether the accumulated motion sequence gives enough evidence to guess.

Respond ONLY with a JSON object — no explanation, no markdown fences, nothing else:
{"context_clues": ["description 1", "description 2", ...], "ready_to_guess": true}

Rules:
- Each clue should be a rich, specific description (up to 15 words) of an action or \
  movement, e.g. "right hand sweeps upward in arc from waist to shoulder height" \
  or "both arms extend outward then pull inward repeatedly as if rowing".
- Reference frame numbers when describing motion changes, \
  e.g. "between Frame 2 and Frame 4, left hand rotates clockwise at wrist".
- "ready_to_guess" is true only when you are highly confident about the charade word.
- Do NOT include your final guess here — only motion/action clues.
- Maximum 8 clues per response (prioritise the most distinctive movements).
- IGNORE and DO NOT mention: clothing, hair, background, furniture, room decor, \
  or any text visible in the environment.
- FOCUS ON: hand gestures, arm/leg/head movements, posture shifts, facial \
  expressions as mime signals, and objects being directly interacted with.
"""

GUESSER_SYSTEM_PROMPT = """\
You are the final guesser in a visual charade guessing game. You will receive:
- Sequential numbered frames (Frame 1, Frame 2, ...) showing a person miming a word/phrase.
- A supplemental list of context clues extracted from those frames.

Your job: produce exactly one short guess (1-5 words) for the charade word or phrase.

Weighting:
- PRIMARY source: the frames themselves — trust what you directly observe in the \
  images above all else. Study the motion sequence, body language, and gestures \
  across all frames as your main evidence.
- SUPPLEMENTAL source: the context clues — use them only to break ties or confirm \
  what you already see in the frames. Discard any clue that contradicts your \
  visual reading of the frames.

Rules:
- Output ONLY the guess word or phrase — nothing else. No explanation, no bullet \
  points, no reasoning, no punctuation other than spaces between words.
- Your entire response must be 1-5 words maximum.
- Be specific: "golden retriever" is better than "dog".
- If the frames clearly show an action, name that action or the thing being mimed.
"""

# ---------------------------------------------------------------------------
# ContextCollection
# ---------------------------------------------------------------------------


@dataclass
class ContextCollection:
    """Accumulates textual clues and raw frames across multiple analyze() calls."""

    text: list[str] = field(default_factory=list)
    binary_data: list[Frame] = field(default_factory=list)

    def append_text(self, clue: str) -> None:
        self.text.append(clue)

    def remove_text(self, clue: str) -> None:
        self.text.remove(clue)

    def append_binary(self, frame: Frame) -> None:
        self.binary_data.append(frame)
        # Enforce rolling window to avoid token overflow
        if len(self.binary_data) > MAX_FRAMES:
            self.binary_data = self.binary_data[-MAX_FRAMES:]

    def remove_binary(self, frame: Frame) -> None:
        self.binary_data.remove(frame)

    def reset(self) -> None:
        self.text.clear()
        self.binary_data.clear()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_context = ContextCollection()
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_to_image_content(frame: Frame) -> dict[str, Any]:
    buf = io.BytesIO()
    frame.image.save(buf, format="JPEG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _frames_to_ordinal_content(frames: list[Frame]) -> list[dict[str, Any]]:
    """Build an interleaved content list with ordinal labels before each frame.

    Produces: [text "Frame 1 of N:", image, text "Frame 2 of N:", image, ...]
    so the model can reference specific frames by number when describing motion.
    """
    total = len(frames)
    content: list[dict[str, Any]] = []
    for i, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {i} of {total}:"})
        content.append(_frame_to_image_content(frame))
    return content


async def _run_extractor(frames: list[Frame]) -> tuple[list[str], bool]:
    """Call AI #1. Returns (new_clues, ready_to_guess)."""
    content: list[dict[str, Any]] = _frames_to_ordinal_content(frames)
    content.append({
        "type": "text",
        "text": (
            "Analyze the frame(s) above. "
            "Extract context clues and decide if you are ready to guess. "
            "Respond with JSON only."
        ),
    })

    response = await _get_client().chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Two-stage JSON parse: direct first, regex fallback for markdown leakage
    data: dict[str, Any] = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return [], False
        else:
            return [], False

    clues: list[str] = [
        c for c in data.get("context_clues", [])
        if isinstance(c, str) and c.strip()
    ]
    ready: bool = bool(data.get("ready_to_guess", False))
    return clues, ready


async def _run_guesser(frames: list[Frame], clues: list[str]) -> str:
    """Call AI #2. Returns the final guess string."""
    content: list[dict[str, Any]] = _frames_to_ordinal_content(frames)

    clue_block = "\n".join(f"- {c}" for c in clues) if clues else "(none)"
    content.append({
        "type": "text",
        "text": (
            f"Context clues gathered so far:\n{clue_block}\n\n"
            "What is the charade word or phrase? Give your single best guess."
        ),
    })

    response = await _get_client().chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=16,
        messages=[
            {"role": "system", "content": GUESSER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Take only the first line and collapse any internal whitespace to a single space,
    # ensuring the string is safe to submit directly to client.guess().
    guess = " ".join(raw.splitlines()[0].split()) if raw else ""
    return guess


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame using a two-AI agentic loop.

    Accumulates frames and context clues across calls. Returns None until the
    extractor AI signals confidence, then fires the guesser AI for a final answer.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string when ready, or None to continue accumulating.
    """
    # Step 1: Add incoming frame to the collection
    _context.append_binary(frame)

    # Step 2: Run the extractor on all current frames
    new_clues, ready_to_guess = await _run_extractor(_context.binary_data)

    # Step 3: Deduplicate and store new context clues
    existing_lower = {c.lower() for c in _context.text}
    for clue in new_clues:
        if clue.lower() not in existing_lower:
            _context.append_text(clue)
            existing_lower.add(clue.lower())

    print(f"  [context] clues={_context.text} ready={ready_to_guess}")

    # Step 4: If the extractor is confident, run the guesser and reset
    if ready_to_guess:
        guess = await _run_guesser(_context.binary_data, _context.text)
        _context.reset()
        return guess if guess else None

    # Step 5: Not ready yet — signal caller to keep accumulating
    return None
