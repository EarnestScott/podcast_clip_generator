"""
AI Podcast Clip Generator + Captioner MVP
"""

# === Imports ===
import os
from pathlib import Path
import whisper
import openai
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import TextClip
import ffmpeg

# TODO: Add any required pip packages (e.g., openai, moviepy, whisper, etc.)
# pip install openai moviepy ffmpeg-python whisper

# === Constants ===
INPUT_FILE = "input/unclog_arteries.mp4"
WHISPER_MODEL = "base"  # or "medium"/"large" for better accuracy
CLIP_MIN_DURATION = 30  # in seconds
CLIP_MAX_DURATION = 90  # in seconds

# === Step 1: Transcription ===
def transcribe_audio(file_path: str) -> dict:
    """
    Uses OpenAI Whisper to transcribe a video/audio file and get timestamped segments.
    """
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(file_path)
    return {
        "text": result["text"],
        "segments": result["segments"]
    }

# === Step 2: Extract Clip Candidates ===
def find_best_clip(transcript_text: str) -> dict:
    """
    Uses GPT-4o to identify the most interesting 30–90 second moment from the transcript.
    Returns start_time, end_time, title, quote.
    """
    prompt = f"""
    Given this podcast transcript, identify the most engaging 30-90 second clip that would work well on social media.
    The clip should be self-contained and interesting even out of context.
    
    Transcript: {transcript_text}
    
    Return a valid JSON object (no markdown formatting) with these fields:
    - start_time (in seconds)
    - end_time (in seconds)
    - title (catchy title for the clip)
    - caption (engaging social media caption)
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",  # Using the correct GPT-4o model identifier
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Clean the response content by removing any markdown formatting
    content = response.choices[0].message.content
    content = content.replace('```json', '').replace('```', '').strip()
    
    return eval(content)  # Parse the JSON response

# === Step 3: Clip the Video ===
def cut_clip(input_file: str, start_time: float, end_time: float, output_file: str):
    """
    Uses ffmpeg to cut the original video file between start_time and end_time.
    Handles both video and audio streams.
    """
    try:
        # Input stream
        stream = ffmpeg.input(input_file)
        
        # Trim both video and audio streams
        video = stream.video.trim(start=start_time, end=end_time).setpts('PTS-STARTPTS')
        audio = stream.audio.filter_('atrim', start=start_time, end=end_time).filter_('asetpts', 'PTS-STARTPTS')
        
        # Combine video and audio streams
        stream = ffmpeg.output(video, audio, output_file)
        
        # Run ffmpeg
        ffmpeg.run(stream, overwrite_output=True)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print('An error occurred:', error_message)
        raise

# === Step 4: Generate Captions ===
def generate_subtitles(segments: list, start_time: float, end_time: float) -> list:
    """
    Given Whisper segments and the desired clip time range, generate subtitle lines.
    """
    clip_segments = []
    for segment in segments:
        if segment['start'] >= start_time and segment['end'] <= end_time:
            # Adjust timestamps relative to clip start
            clip_segments.append({
                'start': segment['start'] - start_time,
                'end': segment['end'] - start_time,
                'text': segment['text']
            })
    return clip_segments

# === Step 5: Burn Captions Into Clip ===
def generate_headline_text(clip_text: str) -> str:
    """
    Generates a punchy, emotionally engaging headline for a social media clip.
    The headline is designed to stop scroll and drive engagement.
    
    Args:
        clip_text (str): The transcript text of the clip
        
    Returns:
        str: A punchy headline under 80 characters
    """
    prompt = f"""
You're a professional video editor for thought-provoking podcast clips on TikTok and Instagram.

Write a headline to overlay at the top of this video clip that:
- Is under 80 characters
- Feels authentic, intelligent, and emotionally resonant
- Sparks curiosity or tension without sounding clickbaity
- Would make a smart, curious viewer stop scrolling
- Avoids slang, emojis, or excessive hype
- Is relevant and specific to the clip content
- Sounds like something from Lex Fridman, Chris Williamson, or Tim Ferriss

Here are a few examples of the tone and structure:
- "Why most people never escape burnout"
- "The surprising truth about ultra-processed food"
- "A better way to build self-discipline"
- "This advice changed how I approach money forever"

Clip content:
{clip_text}

Return ONLY the headline text. Nothing else.
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def add_subtitles_to_clip(video_path: str, subtitles: list, output_path: str):
    """
    Overlay a viral-style headline onto the video using MoviePy.
    """
    video = VideoFileClip(video_path)
    
    # Generate headline from the first subtitle's text
    headline_text = generate_headline_text(subtitles[0]['text'])
    
    # Create headline text clip
    headline = TextClip(
        text=headline_text,
        font='Arial',
        font_size=50,
        color='white',
        stroke_color='black',
        stroke_width=4,
        size=video.size,
        method='caption',
        duration=video.duration
    ).with_position(('center', 'top'))
    
    # Combine video with headline
    final = CompositeVideoClip([video, headline])
    final.write_videofile(output_path)
    
    # Clean up
    video.close()
    final.close()

# === Entry Point ===
def main():
    # Step 1: Transcribe
    transcript_result = transcribe_audio(INPUT_FILE)
    transcript_text = transcript_result["text"]
    segments = transcript_result["segments"]  # [{'start':..., 'end':..., 'text':...}]

    # Step 2: Ask GPT-4 for clip
    clip_info = find_best_clip(transcript_text)
    print("Selected Clip Info:", clip_info)

    start = clip_info["start_time"]
    end = clip_info["end_time"]
    output_clip_path = "output/output_clip.mp4"

    # Step 3: Cut the video
    cut_clip(INPUT_FILE, start, end, output_clip_path)

    # Step 4: Generate subtitles
    subtitles = generate_subtitles(segments, start, end)

    # Step 5: Add subtitles to video
    final_output = "output/final_clip_with_subs.mp4"
    add_subtitles_to_clip(output_clip_path, subtitles, final_output)

    print(f"✅ Final clip created: {final_output}")

if __name__ == "__main__":
    main()

