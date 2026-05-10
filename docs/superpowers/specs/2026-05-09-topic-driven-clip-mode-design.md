# Topic-Driven Clip Mode — Design Spec

**Date:** 2026-05-09
**Status:** Approved

---

## Overview

Add a second mode to the podcast clip generator where users interactively describe specific discussion points they remember from a video. The tool finds those moments in the transcript, cuts them, and outputs clips — instead of GPT autonomously choosing highlights.

**Motivation:** After watching a podcast, users often remember high-value moments but don't want to manually scrub and trim. This mode lets them describe what they remember in plain language and get the clip automatically.

---

## User Flow

1. Tool starts and prompts: `"Mode: (1) Auto-select highlights  (2) Find specific topics  > "`
2. **Mode 1** — existing pipeline runs unchanged.
3. **Mode 2 — Topic mode:**
   - Transcription runs once (same Whisper step).
   - Tool loops: `"Enter a topic description (or press Enter to finish):"`
   - User enters free-text descriptions (e.g., "the part about vitamin d3's impact on mood").
   - For each topic, GPT-4o locates the best-matching discussion window.
   - If no confident match is found: warn and skip.
   - If a match is found: cut the clip, generate subtitles, then prompt: `"Add headline overlay to '[title]'? (y/n)"`
   - Repeat for all topics.

---

## New Function: `find_clip_for_topic()`

```
find_clip_for_topic(transcript_text: str, topic_description: str) -> dict | None
```

- Sends the full transcript + user's topic description to GPT-4o.
- GPT returns a single JSON object:
  ```json
  {
    "start_time": 142.5,
    "end_time": 198.0,
    "title": "Vitamin D3 and mood regulation",
    "clip_text": "...",
    "confidence": "high"
  }
  ```
  `confidence` is one of: `"high"`, `"low"`, `"not_found"`.
- GPT prompt instructs it to:
  - Find the full discussion arc (not just the first mention).
  - Expand the window so the clip is self-contained and makes sense out of context.
  - Return `"not_found"` if nothing matches well.
- Caller logic: if `confidence` is `"low"` or `"not_found"`, print a warning and skip. Otherwise proceed.

**Implementation note:** This uses Option A — GPT-4o with full transcript in context. Option B (embedding-based semantic search over segments) is a viable alternative if transcript length or API cost becomes a concern. Switch to B if transcripts regularly exceed GPT-4o's context window or per-topic cost is too high.

---

## Changes to `main()`

- Add mode selection prompt at startup.
- Mode 1: existing `find_multiple_clips()` path, no changes.
- Mode 2: collect topic list interactively, then for each topic after transcription:
  1. Call `find_clip_for_topic()`.
  2. If no match: warn and skip.
  3. If match: `cut_clip()` → `generate_subtitles()` → optionally `add_subtitles_to_clip()`.
- Transcription result (Whisper) is computed once and reused across all topics.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Topic not found / low confidence | Print warning, skip topic, continue with remaining topics |
| User enters no topics | Exit gracefully with a message |
| ffmpeg / MoviePy error on a clip | Existing error handling applies (raise + print) |

---

## What Doesn't Change

- `transcribe_audio()` — unchanged
- `cut_clip()` — unchanged
- `generate_subtitles()` — unchanged
- `add_subtitles_to_clip()` / `generate_headline_text()` — unchanged
- `INPUT_FILE` constant — unchanged (user still sets this before running)
- `find_multiple_clips()` — unchanged (still used in Mode 1)

---

## Out of Scope

- Changing how the input file is specified (CLI arg, config file, etc.)
- Showing multiple candidate matches for the user to choose from
- Embedding-based search (deferred to Option B if needed)
