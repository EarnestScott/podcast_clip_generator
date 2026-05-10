"""
AI Podcast Clip Generator + Captioner MVP
"""

# === Imports ===
import os
import json
import shutil
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
def find_multiple_clips(transcript_text: str, max_clips: int = 3) -> list:
    """
    Uses GPT-4 to identify multiple highlight-worthy 30-90 second moments from the transcript.
    Returns a list of non-overlapping clips, each with timing, title, and caption information.
    
    Args:
        transcript_text (str): The full transcript text
        max_clips (int): Maximum number of clips to find (default: 3)
        
    Returns:
        list: List of dictionaries containing clip information
    """
    prompt = f"""
    You are a professional video editor for thought-provoking podcast clips. Your task is to identify {max_clips} 
    high-impact moments from this transcript that would work well as standalone short-form videos.
    
    For each clip:
    - Duration should be 30-90 seconds
    - Must be self-contained and engaging even out of context
    - Should contain a complete thought or story arc
    - Must not overlap with other clips
    - Should be emotionally resonant or intellectually stimulating
    
    Return a JSON array of {max_clips} clips, where each clip has:
    - start_time: float (in seconds)
    - end_time: float (in seconds)
    - title: str (engaging title for the clip)
    - clip_text: str (the exact transcript text for this clip)
    - tiktok_caption: str (engaging caption for social media)
    
    Ensure the clips are:
    - Non-overlapping (no time overlap between clips)
    - High-impact (each could stand alone as a viral clip)
    - Varied in content (if possible, cover different topics/angles)
    
    Transcript:
    {transcript_text}
    
    Return ONLY the JSON array, nothing else.
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Clean the response content by removing any markdown formatting
    content = response.choices[0].message.content
    content = content.replace('```json', '').replace('```', '').strip()
    
    return eval(content)  # Parse the JSON response

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
        
        # Combine video and audio streams with explicit codec settings
        stream = ffmpeg.output(
            video, 
            audio, 
            output_file,
            vcodec='libx264',
            acodec='aac',
            preset='medium',
            crf=23,
            movflags='+faststart'
        )
        
        # Run ffmpeg with overwrite
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def add_subtitles_to_clip(video_path: str, subtitles: list, output_path: str):
    """
    Overlay a viral-style headline onto the video using MoviePy.
    """
    try:
        # Load video with explicit duration handling
        video = VideoFileClip(video_path)
        
        # Ensure we have a valid duration
        if video.duration is None:
            raise ValueError(f"Could not determine video duration for {video_path}")
            
        # Generate headline from the first subtitle's text
        headline_text = generate_headline_text(subtitles[0]['text'])
        
        # Create headline text clip with explicit duration
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
        
        # Write with explicit codec and bitrate settings
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            bitrate='5000k',
            audio_bitrate='192k',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            threads=4,
            logger=None
        )
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise
    finally:
        # Ensure cleanup happens even if there's an error
        if 'video' in locals():
            video.close()
        if 'final' in locals():
            final.close()
        if 'headline' in locals():
            headline.close()

# === Entry Point ===
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

if __name__ == "__main__":
    main()

