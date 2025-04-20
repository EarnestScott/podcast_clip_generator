# AI Podcast Clip Generator

This tool automatically generates engaging social media clips from podcast episodes by:
1. Transcribing the audio using OpenAI's Whisper
2. Identifying the most interesting moments using GPT-4
3. Creating a clip with burned-in subtitles

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

1. Place your podcast video file in the project directory
2. Update the `INPUT_FILE` constant in `main.py` to match your video filename
3. Run the script:
```bash
python main.py
```

The script will:
- Transcribe the audio using Whisper
- Identify the most engaging 30-90 second clip
- Create a new video file with burned-in subtitles
- Output the final clip as `final_clip_with_subs.mp4`

## Configuration

You can adjust the following parameters in `main.py`:
- `WHISPER_MODEL`: Choose between "base", "medium", or "large" (larger models are more accurate but slower)
- `CLIP_MIN_DURATION`: Minimum clip duration in seconds (default: 30)
- `CLIP_MAX_DURATION`: Maximum clip duration in seconds (default: 90)

## Output Files

- `output_clip.mp4`: The raw clip without subtitles
- `final_clip_with_subs.mp4`: The final clip with burned-in subtitles

## Notes

- The quality of transcription depends on the Whisper model size:
  - "base": Fastest but least accurate
  - "medium": Good balance of speed and accuracy
  - "large": Most accurate but slowest
- Make sure your input video has clear audio for best results
- The script requires an active internet connection for API calls 