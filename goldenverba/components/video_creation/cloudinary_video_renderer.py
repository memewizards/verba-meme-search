import json
import logging
import cloudinary
import cloudinary.uploader
import cloudinary.api
import time
import time
import cloudinary.utils

import requests
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CloudinaryVideoRenderer:
    def __init__(self, cloudinary_config):
        self.cloudinary_config = cloudinary_config
        cloudinary.config(**cloudinary_config)
        logger.info("CloudinaryVideoRenderer initialized")

    @staticmethod
    def extract_public_id_or_url(url_or_description):
        if url_or_description is None:
            return None
        if url_or_description.startswith('http'):
            # Extract public_id from URL
            parts = url_or_description.split('/')
            public_id_with_ext = parts[-1]
            public_id = public_id_with_ext.rsplit('.', 1)[0]
            return public_id
        else:
            # For non-URL descriptions, try to extract a filename-like part
            import re
            match = re.search(r"'([^']+)'", url_or_description)
            return match.group(1) if match else url_or_description
        
    def create_background_video(self, duration, background_image_public_id):
        logger.info("Creating background video from the uploaded image")
        try:
            # Generate a video by looping the background image to match the total duration
            background_video_url = f"https://res.cloudinary.com/{self.cloudinary_config['cloud_name']}/image/upload/du_{int(duration)},e_loop/{background_image_public_id}"
            logger.debug(f"Background video URL: {background_video_url}")
            return background_video_url
        except Exception as e:
            logger.error(f"Error creating background video: {str(e)}")
            return None

    def create_video_request(self, detailed_schema):
        video_request = []
        total_duration = 0

        for shot_index, shot in enumerate(detailed_schema['shots']):
            logger.info(f"Processing shot {shot_index + 1}")
            
            for dialogue in shot.get('dialogue', []):
                audio_url = dialogue.get('audio_url')
                if audio_url:
                    dialogue_duration = float(dialogue.get('duration', 0))
                    public_id = self.extract_public_id_or_url(audio_url)
                    video_request.append({
                        "overlay": {
                            "resource_type": "video",
                            "public_id": public_id
                        },
                        "start_offset": total_duration,
                        "duration": dialogue_duration
                    })
                    total_duration += dialogue_duration
                else:
                    logger.warning(f"Dialogue audio missing for shot {shot_index + 1}")

        logger.debug(f"Final video request: {video_request}")
        return video_request, total_duration


    def render_video(self, background_video_url, video_request, total_duration):
        logger.info("Rendering final video with audio overlays")
        if not video_request:
            logger.error("Empty video request. Cannot render video.")
            return None

        try:
            transformations = [
                {"w": 1280, "h": 720, "c": "fill"},
                {"du": str(int(total_duration))}
            ]

            for request in video_request:
                overlay_public_id = self.extract_public_id_or_url(request["overlay"]["public_id"])
                if overlay_public_id is not None:
                    transformations.extend([
                        {
                            "l": f"audio:{overlay_public_id}",
                            "so": str(int(request['start_offset'] * 1000)),
                            "du": str(int(request['duration'] * 1000))
                        },
                        {"fl": "layer_apply"}
                    ])
                else:
                    logger.warning(f"Skipping audio overlay due to missing public ID: {request}")

            transformation_str = self.transformations_to_string(transformations)

            print(f"Transformation string: {transformation_str}")

            # Prepare the API request
            timestamp = int(time.time())
            url = f"https://api.cloudinary.com/v1_1/{self.cloudinary_config['cloud_name']}/video/upload"
            data = {
                "file": background_video_url,
                "resource_type": "video",
                "transformation": transformation_str,
                "public_id": f"rendered_video_{timestamp}",
                "timestamp": timestamp
            }

            # Generate the signature using only the required parameters
            params_to_sign = {
                "public_id": data['public_id'],
                "timestamp": data['timestamp'],
                "transformation": data['transformation']
            }
            signature = cloudinary.utils.api_sign_request(params_to_sign, self.cloudinary_config['api_secret'])
            data['signature'] = signature
            data['api_key'] = self.cloudinary_config['api_key']

            logger.debug(f"API request data: {data}")

            # Make the API request
            response = requests.post(url, data=data)
            result = response.json()

            logger.debug(f"Cloudinary API response: {result}")

            # Check for errors in the API response
            if 'error' in result:
                logger.error(f"Cloudinary API error: {result['error']['message']}")
                return None

            # Retrieve the URL of the transformed video
            if 'secure_url' in result:
                video_url = result['secure_url']
                logger.info(f"Video rendered successfully. URL: {video_url}")
                return video_url
            else:
                logger.error("Transformation did not return a video URL.")
                logger.error(f"Full result: {result}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in render_video: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            return None
    def transformations_to_string(self, transformations):
        return '/'.join([','.join([f"{k}_{v}" for k, v in t.items()]) for t in transformations])
        
    def process_schema_for_video(self, detailed_schema):
        logger.info("Starting video processing")
        try:
            video_request, total_duration = self.create_video_request(detailed_schema)
            if not video_request:
                logger.error("Failed to create video request")
                return None

            # Use the uploaded background image's public_id
            background_image_public_id = "whitelayer_eaxhya"  # Remove file extension and folder path
            background_video_url = self.create_background_video(total_duration, background_image_public_id)
            if not background_video_url:
                logger.error("Failed to create background video")
                return None

            # Render final video with audio overlays
            video_url = self.render_video(background_video_url, video_request, total_duration)
            logger.info("Video processing complete")
            return video_url
        except Exception as e:
            logger.error(f"Unexpected error in processing schema: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    cloudinary_config = {
        "cloud_name": "your_cloud_name",
        "api_key": "your_api_key",
        "api_secret": "your_api_secret"
    }

    renderer = CloudinaryVideoRenderer(cloudinary_config)
    
    try:
        with open('detailed_story_schema_with_audio.json', 'r') as f:
            schema = json.load(f)
        
        video_url = renderer.process_schema_for_video(schema)
        if video_url:
            print(f"Generated video URL: {video_url}")
        else:
            print("Failed to generate video")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema file: {str(e)}")
    except FileNotFoundError:
        logger.error("Schema file not found")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")