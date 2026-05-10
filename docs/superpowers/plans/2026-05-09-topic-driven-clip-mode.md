# Topic-Driven Clip Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive topic mode to the podcast clip generator where users describe specific discussion points and the tool finds, cuts, and outputs those clips automatically.

**Architecture:** Add a `find_clip_for_topic()` function that sends the full transcript + user's topic description to GPT-4o and returns the best-matching time window. Update `main()` to prompt for mode selection (auto vs. topic), collect topic descriptions in a loop, and run the existing cut/subtitle/headline pipeline per topic — with a per-clip prompt for the optional headline overlay.

**Tech Stack:** Python 3, OpenAI GPT-4o (`openai` SDK), `unittest.mock` for tests, `pytest` for test runner.

---

## File Map

| File | Change |
|------|--------|
| `main.py` | Add `import json`, `import shutil`; add `find_clip_for_topic()`; refactor `main()` for mode selection + topic loop |
| `tests/test_main.py` | Create — unit tests for `find_clip_for_topic()` |

---

### Task 1: Add `find_clip_for_topic()` with tests (TDD)

**Files:**
- Create: `tests/test_main.py`
- Modify: `main.py` (add function + two imports)

- [ ] **Step 1: Create the tests file with three failing tests**

Create `tests/test_main.py`:

```python
import json
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')
from main import find_clip_for_topic


def _mock_openai(content: str):
    """Helper: returns a mock OpenAI client whose chat completion returns `content`."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = content
    return mock_client


def test_find_clip_high_confidence():
    payload = {
        "start_time": 142.5,
        "end_time": 198.0,
        "title": "Vitamin D3 and mood regulation",
        "clip_text": "Vitamin D3 has significant impacts on mood...",
        "confidence": "high"
    }
    with patch('main.openai.OpenAI', return_value=_mock_openai(json.dumps(payload))):
        result = find_clip_for_topic("...transcript...", "vitamin d3 and mood")

    assert result["confidence"] == "high"
    assert result["start_time"] == 142.5
    assert result["end_time"] == 198.0
    assert result["title"] == "Vitamin D3 and mood regulation"


def test_find_clip_not_found():
    payload = {
        "start_time": 0,
        "end_time": 0,
        "title": "",
        "clip_text": "",
        "confidence": "not_found"
    }
    with patch('main.openai.OpenAI', return_value=_mock_openai(json.dumps(payload))):
        result = find_clip_for_topic("...transcript...", "aliens on mars")

    assert result["confidence"] == "not_found"


def test_find_clip_strips_markdown_fences():
    payload = {
        "start_time": 50.0,
        "end_time": 100.0,
        "title": "Test clip",
        "clip_text": "Some text",
        "confidence": "high"
    }
    wrapped = "```json\n" + json.dumps(payload) + "\n```"
    with patch('main.openai.OpenAI', return_value=_mock_openai(wrapped)):
        result = find_clip_for_topic("...transcript...", "test topic")

    assert result["confidence"] == "high"
    assert result["start_time"] == 50.0
```

- [ ] **Step 2: Run the tests — expect ImportError / AttributeError (function doesn't exist yet)**

```bash
cd /Users/earnestscott/cursorProjects/podcast_clip_generator
pytest tests/test_main.py -v
```

Expected: 3 errors like `ImportError: cannot import name 'find_clip_for_topic' from 'main'`

- [ ] **Step 3: Add `import json`, `import shutil` to `main.py` and add `find_clip_for_topic()`**

At the top of `main.py`, after the existing imports, add:
```python
import json
import shutil
```

After `find_multiple_clips()` (around line 89), add:

```python
# === Step 2b: Find Clip for User-Described Topic ===
def find_clip_for_topic(transcript_text: str, topic_description: str) -> dict:
    """
    Uses GPT-4o to locate the best-matching discussion window for a user-described topic.
    Returns a dict with start_time, end_time, title, clip_text, and confidence.
    confidence is one of: "high", "low", "not_found".

    NOTE (Option A): Sends the full transcript to GPT-4o in a single call.
    Option B alternative: embed the topic description and Whisper segments via
    OpenAI embeddings, find the closest segment by cosine similarity, then expand
    the window outward. Switch to Option B if transcripts regularly exceed GPT-4o's
    context window or per-topic API cost becomes a concern.
    """
    prompt = f"""
You are a professional video editor. A user wants to find a specific discussion in a podcast transcript.

User's topic description: "{topic_description}"

Find the best-matching discussion window in the transcript below. Return the full arc of the
conversation on this topic — not just the first mention. Expand the window so the clip is
self-contained and makes sense out of context.

Return a JSON object with:
- start_time: float (in seconds)
- end_time: float (in seconds)
- title: str (descriptive title for this clip)
- clip_text: str (the transcript text for this segment)
- confidence: str — one of: "high" (clear match), "low" (uncertain), "not_found" (no match)

If no discussion matches the topic, return confidence "not_found" with start_time and end_time as 0.

Transcript:
{transcript_text}

Return ONLY the JSON object, nothing else.
"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    content = content.replace('```json', '').replace('```', '').strip()
    return json.loads(content)
```

- [ ] **Step 4: Run the tests — expect all 3 to pass**

```bash
pytest tests/test_main.py -v
```

Expected output:
```
tests/test_main.py::test_find_clip_high_confidence PASSED
tests/test_main.py::test_find_clip_not_found PASSED
tests/test_main.py::test_find_clip_strips_markdown_fences PASSED

3 passed
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add find_clip_for_topic() with GPT-4o transcript search"
```

---

### Task 2: Refactor `main()` for mode selection and topic pipeline

**Files:**
- Modify: `main.py` — replace `main()` body

- [ ] **Step 1: Replace `main()` with the new implementation**

Replace the entire `main()` function in `main.py` (currently lines 242–270) with:

```python
def main():
    mode = input("Mode: (1) Auto-select highlights  (2) Find specific topics\n> ").strip()

    # Transcribe once — reused by both modes
    transcript_result = transcribe_audio(INPUT_FILE)
    transcript_text = transcript_result["text"]
    segments = transcript_result["segments"]

    if mode == "1":
        # --- Auto mode (unchanged behavior) ---
        clip_infos = find_multiple_clips(transcript_text)
        print("Selected Clip Infos:", clip_infos)

        for clip_info in clip_infos:
            start = clip_info["start_time"]
            end = clip_info["end_time"]
            output_clip_path = f"output/output_clip_{clip_info['title'].replace(' ', '_')}.mp4"

            cut_clip(INPUT_FILE, start, end, output_clip_path)
            subtitles = generate_subtitles(segments, start, end)

            final_output = f"output/final_clip_with_subs_{clip_info['title'].replace(' ', '_')}.mp4"
            add_subtitles_to_clip(output_clip_path, subtitles, final_output)

            print(f"✅ Final clip created: {final_output}")

    elif mode == "2":
        # --- Topic mode ---
        print("Enter topic descriptions one at a time. Press Enter with no text to finish.")
        topics = []
        while True:
            topic = input("Topic > ").strip()
            if not topic:
                break
            topics.append(topic)

        if not topics:
            print("No topics entered. Exiting.")
            return

        for topic in topics:
            print(f"\nSearching for: '{topic}'...")
            result = find_clip_for_topic(transcript_text, topic)

            if result["confidence"] in ("low", "not_found"):
                print(f"⚠️  Could not find a confident match for: '{topic}' — skipping.")
                continue

            title = result["title"]
            start = result["start_time"]
            end = result["end_time"]
            safe_title = title.replace(' ', '_')

            output_clip_path = f"output/output_clip_{safe_title}.mp4"
            cut_clip(INPUT_FILE, start, end, output_clip_path)

            subtitles = generate_subtitles(segments, start, end)

            add_headline = input(f"Add headline overlay to '{title}'? (y/n) ").strip().lower()
            final_output = f"output/final_clip_with_subs_{safe_title}.mp4"

            if add_headline == "y":
                add_subtitles_to_clip(output_clip_path, subtitles, final_output)
            else:
                # No headline requested — copy the raw cut as the final output
                shutil.copy(output_clip_path, final_output)

            print(f"✅ Final clip created: {final_output}")

    else:
        print("Invalid mode. Please enter 1 or 2.")
```

- [ ] **Step 2: Run the existing tests to confirm nothing is broken**

```bash
pytest tests/test_main.py -v
```

Expected: all 3 tests still pass.

- [ ] **Step 3: Smoke test — run in topic mode with your actual video**

```bash
python main.py
```

At the prompt enter `2`, then enter one topic (e.g., "vitamin d3 and mood"), press Enter to finish, and verify:
- GPT locates the clip and prints the title and timestamps
- The clip is cut to `output/output_clip_<title>.mp4`
- You are prompted for headline overlay
- `output/final_clip_with_subs_<title>.mp4` is created and plays correctly

Also verify mode 1 still works by running again and entering `1`.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add interactive topic mode with optional headline overlay"
```
