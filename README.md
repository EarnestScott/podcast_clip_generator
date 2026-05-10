# AI Podcast Clip Generator

This tool automatically generates engaging social media clips from podcast episodes. It supports two modes:

**Auto mode** — GPT-4o identifies the most interesting moments from the full transcript and generates clips automatically.

**Topic mode** — You describe specific discussions you remember (e.g. "the part about vitamin D3 and mood"), and the tool finds those moments, cuts them, and outputs the clips. Great for when you've already watched a podcast and want to save the parts that stood out.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg installed on your system

### FFmpeg Installation
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg website](https://ffmpeg.org/download.html) or use Chocolatey: `choco install ffmpeg`

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd podcast_clip_generator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Place your podcast video file in the `input/` directory
2. Update the `INPUT_FILE` constant in `main.py` to point to your video
3. Run the script:
```bash
python main.py
```

You'll be prompted to choose a mode:

**Mode 1 — Auto-select highlights**
GPT-4o picks the best moments from the transcript and generates up to 3 clips, each with a headline overlay.

**Mode 2 — Find specific topics**
Enter topic descriptions one at a time (press Enter with no text to finish):
```
Topic > the part about vitamin D3 and mood
Topic > how creatine benefits the brain
Topic >
```
For each topic the tool finds the best-matching discussion window. If no confident match is found it warns you and skips. For each matched clip you'll be asked whether to add a headline overlay.

## Configuration

You can adjust the following parameters in `main.py`:
- `WHISPER_MODEL`: Choose between "base", "medium", or "large" (larger models are more accurate but slower)
- `CLIP_MIN_DURATION`: Minimum clip duration in seconds (default: 30)
- `CLIP_MAX_DURATION`: Maximum clip duration in seconds (default: 90)

## Output Files

All output goes to the `output/` directory:

- `output_clip_<title>.mp4` — raw cut segment (intermediate)
- `final_clip_with_subs_<title>.mp4` — final clip, with headline overlay if requested

## Notes

- The quality of transcription depends on the Whisper model size:
  - "base": Fastest but least accurate
  - "medium": Good balance of speed and accuracy
  - "large": Most accurate but slowest
- Make sure your input video has clear audio for best results
- The script requires an active internet connection for API calls 