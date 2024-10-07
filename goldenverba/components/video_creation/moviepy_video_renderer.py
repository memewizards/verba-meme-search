import json
import logging
import requests
import time
import os
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MoviePyVideoRenderer:
    def __init__(self, cloudinary_config):
        self.cloudinary_config = cloudinary_config
        logger.info("MoviePyVideoRenderer initialized")

    @staticmethod
    def extract_filename(url):
        return url.split('/')[-1].split('?')[0]

    def create_background_video(self, duration, background_image_url):
        logger.info("Creating background video from the background image")
        try:
            # Download the background image
            response = requests.get(background_image_url)
            background_image_path = 'background_image.png'
            with open(background_image_path, 'wb') as f:
                f.write(response.content)

            # Create a video clip from the image
            video_clip = ImageClip(background_image_path, duration=duration)
            video_clip = video_clip.set_duration(duration)
            video_clip = video_clip.set_fps(24)
            return video_clip
        except Exception as e:
            logger.error(f"Error creating background video: {str(e)}")
            return None

    def create_audio_clips(self, detailed_schema):
        audio_clips = []
        total_duration = 0
        for shot_index, shot in enumerate(detailed_schema['shots']):
            logger.info(f"Processing shot {shot_index + 1}")
            for dialogue in shot.get('dialogue', []):
                audio_url = dialogue.get('audio_url')
                if audio_url:
                    dialogue_duration = float(dialogue.get('duration', 0))
                    start_time = total_duration
                    # Download the audio file
                    audio_filename = self.extract_filename(audio_url)
                    response = requests.get(audio_url)
                    with open(audio_filename, 'wb') as f:
                        f.write(response.content)
                    # Create an AudioFileClip
                    audio_clip = AudioFileClip(audio_filename).set_start(start_time)
                    audio_clips.append(audio_clip)
                    total_duration += dialogue_duration
                else:
                    logger.warning(f"Dialogue audio missing for shot {shot_index + 1}")
        return audio_clips, total_duration

    def render_video(self, background_video_clip, audio_clips, output_filename='output_video.mp4'):
        logger.info("Rendering final video with audio overlays")
        try:
            # Combine all audio clips
            if not audio_clips:
                logger.error("No audio clips to overlay. Cannot render video.")
                return None
            composite_audio = CompositeAudioClip(audio_clips)
            # Set the audio to the background video
            final_video = background_video_clip.set_audio(composite_audio)
            # Write the video file
            final_video.write_videofile(output_filename, codec='libx264', audio_codec='aac')
            logger.info(f"Video rendered successfully. File saved as: {output_filename}")
            return output_filename
        except Exception as e:
            logger.error(f"Unexpected error in render_video: {str(e)}")
            return None

    def process_schema_for_video(self, detailed_schema):
        logger.info("Starting video processing")
        try:
            # Create audio clips and calculate total duration
            audio_clips, total_duration = self.create_audio_clips(detailed_schema)
            if not audio_clips:
                logger.error("Failed to create audio clips")
                return None

            # Use the background image URL (you can adjust this as needed)
            background_image_url = "https://res.cloudinary.com/{}/image/upload/whitelayer_eaxhya".format(
                self.cloudinary_config['cloud_name']
            )

            # Create the background video clip
            background_video_clip = self.create_background_video(total_duration, background_image_url)
            if not background_video_clip:
                logger.error("Failed to create background video")
                return None

            # Render final video with audio overlays
            video_file = self.render_video(background_video_clip, audio_clips)
            logger.info("Video processing complete")
            return video_file
        except Exception as e:
            logger.error(f"Unexpected error in processing schema: {str(e)}")
            return None
        finally:
            # Clean up downloaded files
            for clip in audio_clips:
                if os.path.exists(clip.filename):
                    os.remove(clip.filename)
            if os.path.exists('background_image.png'):
                os.remove('background_image.png')

