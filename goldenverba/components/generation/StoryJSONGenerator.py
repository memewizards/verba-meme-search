import logging
import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
import openai
import asyncio
from pydub import AudioSegment
from google.cloud import texttospeech
import hashlib

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Import Cloudinary modules
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import regular expressions for filename sanitization
import re

#import cloudinary_video_renderer
from ..video_creation.cloudinary_video_renderer import CloudinaryVideoRenderer
from ..video_creation.moviepy_video_renderer import MoviePyVideoRenderer


load_dotenv()

DEFAULT_VOICE_SETTINGS = {
    "language_code": "en-US",
    "voice_name": "en-US-Wavenet-D",  # Choose a default voice
    "pitch": 0.0,
    "speaking_rate": 1.0
}

# Define Pydantic model for dialogue lines
class DialogueLine(BaseModel):
    character: str = Field(description="Name of the character speaking")
    line: str = Field(description="The line of dialogue spoken by the character")
    audio_file: Optional[str] = Field(default=None, description="Local filename of the TTS audio")
    audio_url: Optional[str] = Field(default=None, description="Cloudinary URL of the TTS audio")
    duration: Optional[float] = Field(default=None, description="Duration of the audio in seconds")

# Define Pydantic models for the initial schema with dialogue
class InitialShot(BaseModel):
    shot_number: int = Field(ge=1, description="Number of the shot in the sequence")
    description: str = Field(description="Description of the shot")
    dialogue: List[DialogueLine] = Field(description="List of dialogue lines in the shot")
    images: List[str] = Field(description="List of image descriptions")
    sound_effects: List[str] = Field(description="List of sound effect descriptions")

class InitialStorySchema(BaseModel):
    shots: List[InitialShot] = Field(description="List of shots in the story")

# Define Pydantic models for the detailed schema with dialogue
class ImageLayer(BaseModel):
    layer: Literal["Background", "Subject", "Foreground"] = Field(description="Layer type")
    duration: float = Field(ge=0.1, description="Duration of the image layer in seconds")
    description: str = Field(description="Description of the image content")

class SoundEffect(BaseModel):
    layer: Literal["Foley", "Ambiance", "Music"] = Field(description="Layer type")
    duration: float = Field(ge=0.1, description="Duration of the sound effect in seconds")
    description: str = Field(description="Description of the sound effect")

class DetailedShot(BaseModel):
    shot_number: int = Field(ge=1, description="Number of the shot in the sequence")
    description: str = Field(description="Description of the shot")
    dialogue: List[DialogueLine] = Field(description="List of dialogue lines in the shot")
    duration: float = Field(ge=0.1, description="Duration of the shot in seconds")
    images: List[ImageLayer] = Field(description="List of image layers in the shot")
    sound_effects: List[SoundEffect] = Field(description="List of sound effects in the shot")

class DetailedStorySchema(BaseModel):
    shots: List[DetailedShot] = Field(description="List of shots in the story")

class StoryJSONGenerator(Generator):

    def __init__(self):
        super().__init__()
        self.name = "StoryJSONGenerator"
        self.description = "Generator for creating detailed story JSON schemas for animated videos"
        self.requires_library = ["openai", "cloudinary"]
        self.requires_env = [
            "OPENAI_API_KEY",
            "CLOUDINARY_CLOUD_NAME",
            "CLOUDINARY_API_KEY",
            "CLOUDINARY_API_SECRET"
        ]
        self.streamable = True
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4")
        self.context_window = 10000

        # Cloudinary configuration
        self.cloudinary_config = {
            "cloud_name": os.getenv('CLOUDINARY_CLOUD_NAME'),
            "api_key": os.getenv('CLOUDINARY_API_KEY'),
            "api_secret": os.getenv('CLOUDINARY_API_SECRET'),
            "secure": True
        }
        
                # Add a class attribute for video rendering control
        self.render_video = os.getenv('RENDER_VIDEO_WITH_CLOUDINARY', 'False').lower() == 'true'


        # Initialize Cloudinary
        cloudinary.config(**self.cloudinary_config)

        # Initialize CloudinaryVideoRenderer
        self.video_renderer = MoviePyVideoRenderer(self.cloudinary_config)
        
        # Initialize other necessary components
        self.initialize_components()

    def initialize_components(self):
        # Initialize CloudinaryVideoRenderer
        self.video_renderer = CloudinaryVideoRenderer(self.cloudinary_config)
        self.video_renderer = MoviePyVideoRenderer(self.cloudinary_config)

    def load_character_voice_mapping(self, character_names):
        # Load character voice mapping from a JSON file
        voice_mapping_file = "character_voices.json"
        if os.path.exists(voice_mapping_file):
            with open(voice_mapping_file, "r") as f:
                self.CHARACTER_VOICE_MAPPING = json.load(f)
            logging.info("Character voice mapping loaded successfully.")
        else:
            self.CHARACTER_VOICE_MAPPING = {}
            logging.warning(f"Character voice mapping file '{voice_mapping_file}' not found. Starting with an empty mapping.")

        # Log the loaded mappings
        for character, settings in self.CHARACTER_VOICE_MAPPING.items():
            logging.debug(f"Loaded voice settings for '{character}': {settings}")

        # Assign default voice settings to any new characters
        for character in character_names:
            if character not in self.CHARACTER_VOICE_MAPPING:
                self.CHARACTER_VOICE_MAPPING[character] = DEFAULT_VOICE_SETTINGS.copy()
                logging.info(f"Assigned default voice settings to new character '{character}'.")


    async def generate_stream(self, queries: List[str], context: List[str], conversation: dict = None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            openai.api_base = base_url

        if "OPENAI_API_TYPE" in os.environ:
            openai.api_type = os.getenv("OPENAI_API_TYPE")
        if "OPENAI_API_BASE" in os.environ:
            openai.api_base = os.getenv("OPENAI_API_BASE")
        if "OPENAI_API_VERSION" in os.environ:
            openai.api_version = os.getenv("OPENAI_API_VERSION")

        user_story_idea = queries[0]
        context_text = "\n".join(context)  # Join all context items into a single string

        # Step 1: Generate initial schema
        initial_schema = await self.generate_initial_schema(user_story_idea, context_text)

        # Save and yield initial schema
        initial_schema_json = initial_schema.model_dump()
        with open("initial_story_schema.json", "w") as f:
            json.dump(initial_schema_json, f, indent=2)

        yield {
            "message": f"Initial story schema generated:\n\n{json.dumps(initial_schema_json, indent=2)}",
            "finish_reason": "",
        }

        # Step 2: Generate detailed schema
        detailed_schema = await self.generate_detailed_schema(initial_schema, context_text)

        # Extract character names from the detailed schema
        character_names = self.extract_character_names(detailed_schema)

        # Load character voice mapping and assign default voice settings
        self.load_character_voice_mapping(character_names)

        # Save and yield detailed schema
        detailed_schema_json = detailed_schema.model_dump()
        with open("detailed_story_schema.json", "w") as f:
            json.dump(detailed_schema_json, f, indent=2)

        yield {
            "message": f"Detailed story schema generated:\n\n{json.dumps(detailed_schema_json, indent=2)}",
            "finish_reason": "",
        }

        # Step 3: Process dialogues for TTS
        self.process_dialogues_for_tts(detailed_schema)

        # Save updated detailed schema with audio files and URLs
        updated_detailed_schema_json = detailed_schema.model_dump()
        with open("detailed_story_schema_with_audio.json", "w") as f:
            json.dump(updated_detailed_schema_json, f, indent=2)

        yield {
            "message": f"Detailed story schema updated with audio files:\n\n{json.dumps(updated_detailed_schema_json, indent=2)}",
            "finish_reason": "",
        }
        print("StoryJSONGenerator: Reached video rendering decision point")
        print(f"StoryJSONGenerator: self.render_video is {self.render_video}")
        print(f"StoryJSONGenerator: video_renderer type: {type(self.video_renderer)}")
        print(f"StoryJSONGenerator: video_renderer config: {self.video_renderer.cloudinary_config}")

        if self.render_video:
            print("StoryJSONGenerator: Starting video rendering process")
            try:
                # Use the updated schema with audio for video rendering
                print(f"StoryJSONGenerator: Type of updated_detailed_schema_json: {type(updated_detailed_schema_json)}")
                print(f"StoryJSONGenerator: Content of updated_detailed_schema_json (truncated): {json.dumps(updated_detailed_schema_json, indent=2)[:500]}...")

                video_url = self.video_renderer.process_schema_for_video(updated_detailed_schema_json)
                print(f"StoryJSONGenerator: Video rendering complete. URL: {video_url}")
                yield {
                    "message": f"Video generated: {video_url}",
                    "finish_reason": "",
                }
            except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    print(f"StoryJSONGenerator: Error during video rendering: {str(e)}")
                    print(f"StoryJSONGenerator: Full error traceback:\n{error_traceback}")
                    yield {
                        "message": f"Error during video rendering: {str(e)}",
                        "finish_reason": "error",
                    }
                
            print("StoryJSONGenerator: Skipping timeline commands generation")
            return  # Exit the generator function here
        else:
            print("StoryJSONGenerator: Video rendering skipped")
            yield {
                "message": "Video rendering skipped.",
                "finish_reason": "",
            }

            print("StoryJSONGenerator: Proceeding to timeline commands generation")
            # Step 4: Generate timeline commands
            timeline_commands = self.video_idea_to_timeline_commands(detailed_schema)
            
            # Save timeline commands to JSON file
            with open("timeline_commands.json", "w") as f:
                json.dump(timeline_commands, f, indent=2)

            yield {
                "message": f"Timeline commands generated:\n\n{json.dumps(timeline_commands, indent=2)}",
                "finish_reason": "stop",
            }


    async def generate_initial_schema(self, user_story_idea, context_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a video production AI assistant. I'm going to describe to you a video that I want to make, and it's an animated video. "
                    "I want you to generate a full schema for every shot, including images, sound effects, and dialogues.\n\n"
                    "For the dialogues:\n"
                    "- Represent each line of dialogue as an object containing the 'character' and their 'line'.\n"
                    "- The 'dialogue' field should be a list of these dialogue objects.\n\n"
                    "Please ensure your response is a valid JSON object that conforms to the specified format."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nStory idea:\n{user_story_idea}",
            },
        ]

        functions = [
            {
                "name": "generate_story_schema",
                "description": "Generate an initial story schema",
                "parameters": InitialStorySchema.model_json_schema(),
            }
        ]

        function_call = {"name": "generate_story_schema"}

        chat_completion_arguments = {
            "model": self.model_name,
            "messages": messages,
            "functions": functions,
            "function_call": function_call,
            "temperature": 0.7,
        }
        if openai.api_type == "azure":
            chat_completion_arguments["deployment_id"] = self.model_name

        completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

        function_response = completion.choices[0].message.function_call.arguments
        parsed_schema = InitialStorySchema.model_validate_json(function_response)

        return parsed_schema

    async def generate_detailed_schema(self, initial_schema, context_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "Now, I want you to break down the initial schema into a more detailed version. This new schema should cover specific details about every shot, image layer, and sound effect.\n\n"
                    "When representing dialogues:\n"
                    "- Use the 'dialogue' field as a list of dialogue lines.\n"
                    "- Each dialogue line should be an object with 'character' and 'line' fields.\n\n"
                    "When designing shots for this video:\n"
                    "1. Include at least 3 image layers per shot: at least 1 background image to set the location, at least 1 subject of the shot, and at least 1 foreground layer to enhance the scene.\n"
                    "2. Include at least 2 sound layers per shot: at least 1 Foley to emphasize the scene actions and at least 1 ambiance or music to carry the emotional tone.\n"
                    "3. Always describe the shot type so the artists know how to frame and compose the shot.\n\n"
                    "Please ensure your response is a valid JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Here's the initial schema to expand upon:\n{initial_schema.model_dump_json(indent=2)}"
                ),
            },
        ]

        functions = [
            {
                "name": "generate_detailed_story_schema",
                "description": "Generate a detailed story schema",
                "parameters": DetailedStorySchema.model_json_schema(),
            }
        ]

        function_call = {"name": "generate_detailed_story_schema"}

        chat_completion_arguments = {
            "model": self.model_name,
            "messages": messages,
            "functions": functions,
            "function_call": function_call,
            "temperature": 0.7,
        }
        if openai.api_type == "azure":
            chat_completion_arguments["deployment_id"] = self.model_name

        completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

        function_response = completion.choices[0].message.function_call.arguments
        parsed_schema = DetailedStorySchema.model_validate_json(function_response)

        return parsed_schema

    async def generate(self, queries: List[str], context: List[str], conversation: dict = None):
        result = None
        async for res in self.generate_stream(queries, context, conversation):
            result = res
        return result

    def process_dialogues_for_tts(self, detailed_schema):
        client = texttospeech.TextToSpeechClient()

        for shot in detailed_schema.shots:
            for dialogue_line in shot.dialogue:
                character = dialogue_line.character.strip()
                line = dialogue_line.line

                # Get voice settings, assign default if missing
                voice_settings = self.CHARACTER_VOICE_MAPPING.get(character)
                if not voice_settings:
                    self.handle_missing_voice_settings(character)
                    voice_settings = self.CHARACTER_VOICE_MAPPING[character]

                # Fallback if language_code or voice_name is missing
                if not voice_settings.get("language_code") or not voice_settings.get("voice_name"):
                    logging.warning(f"Missing language_code or voice_name for '{character}'. Using default settings.")
                    voice_settings = DEFAULT_VOICE_SETTINGS.copy()
                    self.CHARACTER_VOICE_MAPPING[character] = voice_settings

                language_code = voice_settings.get("language_code")
                voice_name = voice_settings.get("voice_name")

                # Extract the first four words from the dialogue line
                first_four_words = '_'.join(line.split()[:4])
                # Remove any characters that are not letters, numbers, or underscores
                first_four_words = re.sub(r'[^A-Za-z0-9_]', '', first_four_words)
                # Limit the length of the first_four_words to prevent overly long filenames
                first_four_words = first_four_words[:50]

                # Generate a hash to ensure filename uniqueness
                filename_hash = hashlib.md5(f"{character}_{line}".encode('utf-8')).hexdigest()[:8]

                # **Sanitize the filename to remove spaces**
                filename_base = f"{character.replace(' ', '_')}_{first_four_words}_{filename_hash}.mp3"
                output_filename = os.path.join("audio", filename_base)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

                # Check if audio file already exists
                if os.path.exists(output_filename):
                    logging.info(f"Using cached audio for '{character}': {output_filename}")
                    # Get audio duration
                    audio_duration = self.get_audio_duration(output_filename)
                else:
                    # Prepare synthesis input
                    synthesis_input = texttospeech.SynthesisInput(text=line)

                    # Prepare voice parameters
                    voice = texttospeech.VoiceSelectionParams(
                        language_code=language_code,
                        name=voice_name
                    )

                    # Prepare audio configuration
                    audio_config = texttospeech.AudioConfig(
                        audio_encoding=texttospeech.AudioEncoding.MP3,
                        pitch=voice_settings.get("pitch", 0.0),
                        speaking_rate=voice_settings.get("speaking_rate", 1.0)
                    )

                    # Perform TTS synthesis
                    try:
                        response = client.synthesize_speech(
                            input=synthesis_input,
                            voice=voice,
                            audio_config=audio_config
                        )
                    except Exception as e:
                        logging.error(f"Error synthesizing speech for '{character}': {e}")
                        continue  # Skip this dialogue line

                    # Save the audio to a file
                    with open(output_filename, "wb") as out:
                        out.write(response.audio_content)
                        logging.info(f"Audio content for '{character}' written to file {output_filename}")

                    # Get audio duration
                    audio_duration = self.get_audio_duration(output_filename)

                # Upload the audio file to Cloudinary
                try:
                    # Use a deterministic public ID based on the sanitized filename without extension
                    public_id = os.path.splitext(filename_base)[0]
                    upload_result = cloudinary.uploader.upload(
                        output_filename,
                        resource_type='video',  # Use 'video' for audio files
                        public_id=public_id,
                        unique_filename=False,
                        overwrite=False  # Do not overwrite if it exists
                    )
                    cloudinary_url = upload_result['secure_url']
                    logging.info(f"Uploaded audio file to Cloudinary: {cloudinary_url}")

                    # Assign local filename, Cloudinary URL, and duration to dialogue line
                    dialogue_line.audio_file = filename_base  # Local filename to be used in timeline commands
                    dialogue_line.audio_url = cloudinary_url  # URL for the client to download
                    dialogue_line.duration = audio_duration

                    # Delete the local audio file after successful upload
                    try:
                        os.remove(output_filename)
                        logging.info(f"Deleted local audio file: {output_filename}")
                    except Exception as e:
                        logging.error(f"Error deleting local audio file '{output_filename}': {e}")

                except Exception as e:
                    logging.error(f"Error uploading audio to Cloudinary for '{character}': {e}")
                    # Optionally, you can decide whether to delete the file if upload fails
                    continue  # Skip this dialogue line


    def get_audio_duration(self, file_path):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000.0  # pydub calculates in milliseconds
        return duration_seconds

    def handle_missing_voice_settings(self, character):
        logging.warning(f"No voice settings found for character: {character}. Assigning default voice settings.")
        self.CHARACTER_VOICE_MAPPING[character] = DEFAULT_VOICE_SETTINGS.copy()

    def extract_character_names(self, detailed_schema):
        character_names = set()
        for shot in detailed_schema.shots:
            for dialogue_line in shot.dialogue:
                character_names.add(dialogue_line.character.strip())
        return character_names
    
    

    def video_idea_to_timeline_commands(self, video_idea):
        logging.info("Starting video_idea_to_timeline_commands method")
        commands = {"message": {"actions": []}}
        current_time = 0.0
        total_commands = 0  # Initialize total_commands

        # Get Cloudinary cloud name from environment variable
        cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')

        # Initialize track assignment
        # Video tracks: Background=1, Subject=2, Foreground=3
        video_track_mapping = {"Background": 1, "Subject": 2, "Foreground": 3}
        # Audio tracks: start at 1, dynamically assigned
        audio_next_track = 1

        # Keep track of audio track usage, mapping to end times
        audio_track_end_times = {}
        def get_transform(layer_name, shot_number, template_name="chad_vs_soyjak_2_panels"):
            
            # Template for the transform of the image layers
            # This could be loaded from a configuration file or database
            templates = {
                    "default": {
                        "Background": {"position": {"x": 0, "y": 0}, "scale": 1.0, "rotation": 0},
                    "Subject": {"position": {"x": 0, "y": 0}, "scale": 1.0, "rotation": 0},
                    "Foreground": {"position": {"x": 0, "y": 0}, "scale": 1.0, "rotation": 0},
                },
                "split_screen": {
                    "Background": {"position": {"x": 0, "y": 0}, "scale": 1.0, "rotation": 0},
                    "Subject": {"position": {"x": -320, "y": 0}, "scale": 0.5, "rotation": 0},
                    "Foreground": {"position": {"x": 320, "y": 0}, "scale": 0.5, "rotation": 0},
                },
                "chad_vs_soyjak_2_panels": {
                    "Background": {"position": {"x": 0, "y": 0}, "scale": 1.0, "rotation": 0},
                    "Subject": {"position": {"x": -320, "y": 0}, "scale": 0.5, "rotation": 0},
                    "Foreground": {"position": {"x": 320, "y": 0}, "scale": 0.5, "rotation": 0},
                },
                # Add more templates as needed
            }
        
            return templates.get(template_name, templates["default"]).get(layer_name, templates["default"][layer_name])

        logging.info(f"Processing video idea with {len(video_idea.shots)} shots")
        for shot_index, shot in enumerate(video_idea.shots):
            logging.info(f"Processing shot {shot_index + 1}")

            shot_duration = float(shot.duration)
            


                    # Image layers
            for layer_name in ["Background", "Subject", "Foreground"]:
                images = [img for img in shot.images if img.layer == layer_name]
                for image in images:
                    clip_path = f"{layer_name.lower()}_{shot.shot_number}.png"
                    cloudinary_url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/video/{clip_path}"
                    
                    # Get the transform for this layer
                    transform = get_transform(layer_name, shot.shot_number)
                    
                    commands["message"]["actions"].append(
                        {
                            "action": "insert_clip",
                            "clip": {
                                "clipPath": clip_path,
                                "url": cloudinary_url,
                                "length": float(image.duration),
                                "track": video_track_mapping[layer_name],
                                "start": current_time,
                                "speed": 1.0,
                                "assetType": "video",
                                "transform": transform  # Add the transform here
                            },
                        }
                    )
                    total_commands += 1
                    logging.info(f"Added {layer_name} image for shot {shot.shot_number} to video track {video_track_mapping[layer_name]}")

            # Sound effects
            for sound in shot.sound_effects:
                clip_path = f"{sound.layer.lower()}_{shot.shot_number}.wav"
                cloudinary_url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/video/upload/{clip_path}"

                # Assign audio track dynamically
                assigned_track = None
                for track, end_time in audio_track_end_times.items():
                    if current_time >= end_time:
                        assigned_track = track
                        break
                if assigned_track is None:
                    # Assign a new track
                    assigned_track = audio_next_track
                    audio_next_track += 1
                # Update end time for the track
                audio_track_end_times[assigned_track] = current_time + float(sound.duration)
                commands["message"]["actions"].append(
                    {
                        "action": "insert_clip",
                        "clip": {
                            "clipPath": clip_path,
                            "url": cloudinary_url,
                            "length": float(sound.duration),
                            "track": assigned_track,
                            "start": current_time,
                            "speed": 1.0,
                            "assetType": "audio",
                        },
                    }
                )
                total_commands += 1
                logging.info(f"Added {sound.layer} sound for shot {shot.shot_number} to audio track {assigned_track}")

            # Dialogue audio files
            dialogue_start_time = current_time
            for dialogue_line in shot.dialogue:
                if dialogue_line.audio_file and dialogue_line.duration:
                    clip_path = dialogue_line.audio_file  # Assuming audio_file is the filename
                    cloudinary_url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/video/upload/{clip_path}"

                    # Assign audio track dynamically
                    assigned_track = None
                    for track, end_time in audio_track_end_times.items():
                        if dialogue_start_time >= end_time:
                            assigned_track = track
                            break
                    if assigned_track is None:
                        # Assign a new track
                        assigned_track = audio_next_track
                        audio_next_track += 1
                    # Update end time for the track
                    audio_track_end_times[assigned_track] = dialogue_start_time + dialogue_line.duration
                    commands["message"]["actions"].append(
                        {
                            "action": "insert_clip",
                            "clip": {
                                "clipPath": clip_path,
                                "url": cloudinary_url,
                                "length": dialogue_line.duration,
                                "track": assigned_track,
                                "start": dialogue_start_time,
                                "speed": 1.0,
                                "assetType": "audio",
                            },
                        }
                    )
                    dialogue_start_time += dialogue_line.duration
                    total_commands += 1
                    logging.info(f"Added dialogue audio for character '{dialogue_line.character}' in shot {shot.shot_number} to audio track {assigned_track}")
                else:
                    logging.warning(f"No audio file or duration for dialogue line: {dialogue_line.line}")

            current_time += shot_duration
            logging.info(f"Finished processing shot {shot_index + 1}")

        logging.info("Finished processing all shots")

        # Convert commands to JSON string
        timeline_commands_json = json.dumps(commands)
        logging.info(f"Generated timeline commands JSON (first 100 chars): {timeline_commands_json[:100]}...")

        # Save timeline commands to file
        file_path = os.path.join(os.getcwd(), "timeline_commands.json")
        logging.info(f"Attempting to save timeline commands to: {file_path}")

        with open(file_path, "w") as f:
            f.write(timeline_commands_json)
        logging.info(f"Timeline commands successfully saved to: {file_path}")

        # Verify file was created and has content
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logging.info(f"Verified: File exists at {file_path} with size {file_size} bytes")
        else:
            logging.error(f"File was not created at {file_path}")

        logging.info(f"Total commands generated: {total_commands}")

        return timeline_commands_json
