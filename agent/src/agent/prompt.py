"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

from core import Frame

# ---------------------------------------------------------------------------
# System prompt — tweak this to improve your agent's guessing ability.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are playing a visual guessing game. You will receive a screenshot from a
live camera feed. Your goal is to identify what is being shown as quickly and
accurately as possible.

Rules:
- Give your best guess as a short, specific answer (1-5 words).
- If you're not confident enough yet, respond with exactly "SKIP".
- Be specific: "golden retriever" is better than "dog".
- You only get to see one frame at a time, so make it count.
"""


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    This is the core function you should customize. The default
    implementation is a simple placeholder that always skips.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    import os
    import io
    import base64
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    buf = io.BytesIO()
    frame.image.save(buf, format="JPEG")
    image_data = base64.standard_b64encode(buf.getvalue()).decode()

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": "What is this? Give your best guess."},
            ]},
        ],
    )
    answer = response.choices[0].message.content.strip()
    return None if answer == "SKIP" else answer
