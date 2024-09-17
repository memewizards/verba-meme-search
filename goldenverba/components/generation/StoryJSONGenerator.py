import logging
import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
import openai
import asyncio
from pydub import AudioSegment
from google.cloud import texttospeech

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

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
    audio_file: Optional[str] = Field(default=None, description="File path to the TTS audio of the line")
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
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = True
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4")
        self.context_window = 10000



    def load_character_voice_mapping(self, character_names):
        # Load character voice mapping from a JSON file
        voice_mapping_file = "character_voices.json"
        if os.path.exists(voice_mapping_file):
            with open(voice_mapping_file, "r") as f:
                self.CHARACTER_VOICE_MAPPING = json.load(f)
        else:
            self.CHARACTER_VOICE_MAPPING = {}
            logging.warning(f"Character voice mapping file '{voice_mapping_file}' not found. Starting with an empty mapping.")

        # Assign default voice settings to any new characters
        for character in character_names:
            if character not in self.CHARACTER_VOICE_MAPPING:
                self.CHARACTER_VOICE_MAPPING[character] = DEFAULT_VOICE_SETTINGS.copy()

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

        # Save updated detailed schema with audio files
        updated_detailed_schema_json = detailed_schema.model_dump()
        with open("detailed_story_schema_with_audio.json", "w") as f:
            json.dump(updated_detailed_schema_json, f, indent=2)

        yield {
            "message": f"Detailed story schema updated with audio files:\n\n{json.dumps(updated_detailed_schema_json, indent=2)}",
            "finish_reason": "",
        }

        # Step 4: Generate timeline commands
        try:
            logging.info("Starting Step 4: Parsing detailed schema and generating timeline commands")
            timeline_commands = self.video_idea_to_timeline_commands(detailed_schema)
            logging.info(f"Generated timeline commands (first 100 chars): {timeline_commands[:100]}...")

            yield {
                "message": f"Timeline commands generated and saved to 'timeline_commands.json':\n\n{timeline_commands}",
                "finish_reason": "stop",
            }
        except Exception as e:
            logging.error(f"Error in generate_stream: {str(e)}")
            logging.error(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            yield {
                "message": f"Error generating timeline commands: {str(e)}",
                "finish_reason": "error",
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
        from google.cloud import texttospeech
        import hashlib

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

                # Ensure language code and voice name are available
                language_code = voice_settings.get("language_code")
                voice_name = voice_settings.get("voice_name")

                if not language_code or not voice_name:
                    logging.error(f"Missing language code or voice name for character '{character}'. Skipping dialogue line.")
                    continue

                # Generate audio file path
                filename_hash = hashlib.md5(f"{character}_{line}".encode('utf-8')).hexdigest()
                output_filename = os.path.join("audio", f"{character}_{filename_hash}.mp3")

                # Check if audio file already exists
                if os.path.exists(output_filename):
                    logging.info(f"Using cached audio for '{character}': {output_filename}")
                    # Get audio duration
                    audio_duration = self.get_audio_duration(output_filename)
                    # Assign audio file and duration to dialogue line
                    dialogue_line.audio_file = output_filename
                    dialogue_line.duration = audio_duration
                    continue

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

                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

                # Save the audio to a file
                with open(output_filename, "wb") as out:
                    out.write(response.audio_content)
                    logging.info(f"Audio content for '{character}' written to file {output_filename}")

                # Get audio duration
                audio_duration = self.get_audio_duration(output_filename)

                # Assign audio file and duration to dialogue line
                dialogue_line.audio_file = output_filename
                dialogue_line.duration = audio_duration


    def get_audio_duration(self, file_path):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000.0  # pydub calculates in milliseconds
        return duration_seconds

    def handle_missing_voice_settings(self, character):
        logging.warning(f"No voice settings found for character: {character}. Assigning default voice settings.")
        self.CHARACTER_VOICE_MAPPING[character] = self.DEFAULT_VOICE_SETTINGS.copy()
        
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

        logging.info(f"Processing video idea with {len(video_idea.shots)} shots")
        for shot_index, shot in enumerate(video_idea.shots):
            logging.info(f"Processing shot {shot_index + 1}")

            shot_duration = float(shot.duration)

            # Image layers
            for layer_name in ["Background", "Subject", "Foreground"]:
                images = [img for img in shot.images if img.layer == layer_name]
                for image in images:
                    commands["message"]["actions"].append(
                        {
                            "action": "insert_clip",
                            "clip": {
                                "clipPath": f"/path/to/{layer_name.lower()}_{shot.shot_number}.png",
                                "length": float(image.duration),
                                "track": {"Background": 1, "Subject": 2, "Foreground": 3}[layer_name],
                                "start": current_time,
                                "speed": 1.0,
                            },
                        }
                    )
                    total_commands += 1
                    logging.info(f"Added {layer_name} image for shot {shot.shot_number}")

            # Sound effects
            for i, sound in enumerate(shot.sound_effects):
                commands["message"]["actions"].append(
                    {
                        "action": "insert_clip",
                        "clip": {
                            "clipPath": f"/path/to/{sound.layer.lower()}_{shot.shot_number}.wav",
                            "length": float(sound.duration),
                            "track": 4 + i,
                            "start": current_time,
                            "speed": 1.0,
                        },
                    }
                )
                total_commands += 1
                logging.info(f"Added {sound.layer} sound for shot {shot.shot_number}")

            # Dialogue audio files
            dialogue_start_time = current_time
            for dialogue_line in shot.dialogue:
                if dialogue_line.audio_file and dialogue_line.duration:
                    commands["message"]["actions"].append(
                        {
                            "action": "insert_clip",
                            "clip": {
                                "clipPath": dialogue_line.audio_file,
                                "length": dialogue_line.duration,
                                "track": 5,  # Assuming track 5 is for dialogue
                                "start": dialogue_start_time,
                                "speed": 1.0,
                            },
                        }
                    )
                    dialogue_start_time += dialogue_line.duration
                    total_commands += 1
                    logging.info(f"Added dialogue audio for character '{dialogue_line.character}' in shot {shot.shot_number}")
                else:
                    logging.warning(f"No audio file or duration for dialogue line: {dialogue_line.line}")

            current_time += shot_duration
            logging.info(f"Finished processing shot {shot_index + 1}")

        logging.info("Finished processing all shots")

        # Convert commands to JSON string
        timeline_commands_json = json.dumps(commands, indent=2)
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
