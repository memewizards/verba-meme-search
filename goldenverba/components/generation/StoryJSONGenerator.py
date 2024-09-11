import logging
import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
import openai
import asyncio

load_dotenv()

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

    async def generate_stream(self, queries: list[str], context: list[str], conversation: dict = None):
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

        # Step 1: Generate initial schema
        initial_system_prompt = """You are a video production AI assistant. I'm going to describe to you a video that I want to make, and it's an animated video. I want you to generate a full schema for every shot, including images and sound effects. Please provide your response in the following JSON format:

{
  "shots": [
    {
      "shot_number": "number",
      "description": "string",
      "images": [
        "string"
      ],
      "sound_effects": [
        "string"
      ]
    }
  ]
}

Please ensure your response is a valid JSON object. When you're ready, I'll give you my story idea, and then you can create the JSON schema."""
        
        initial_schema = await self._generate_response(initial_system_prompt, user_story_idea)
        
        # Save initial schema to file
        with open("initial_story_schema.json", "w") as f:
            f.write(initial_schema)

        yield {
            "message": "Initial story schema generated and saved to 'initial_story_schema.json'",
            "finish_reason": "",
        }

        # Step 2: Generate detailed schema
        detailed_system_prompt = """Now, I want you to break down the initial schema into a more detailed version. This new schema should cover specific details about every shot, image layer, and sound effect. Please provide your response in the following JSON format:

{
  "shots": [
    {
      "shot_number": "number",
      "shot_type": "string",
      "description": "string",
      "duration": "number",
      "images": [
        {
          "layer": "string",
          "description": "string",
          "duration": "number"
        }
      ],
      "sound_effects": [
        {
          "layer": "string",
          "description": "string",
          "duration": "number",
          "volume_db": "number"
        }
      ],
      "camera_movement": "string"
    }
  ]
}

When designing shots for this video:
1. Include at least 3 image layers per shot: at least 1 background image to set the location, at least 1 subject of the shot, and at least 1 foreground layer to enhance the scene (like a vignette or raindrops on alpha).
2. Include at least 2 sound layers per shot: at least 1 Foley to emphasize the scene actions and at least 1 ambiance or music to carry the emotional tone.
3. Always describe the shot type so the artists know how to frame and compose the shot.

Please ensure your response is a valid JSON object. Here's the initial schema to expand upon:

{InitialSchema}"""

        detailed_prompt = detailed_system_prompt.replace("{InitialSchema}", initial_schema)
        detailed_schema = await self._generate_response(detailed_prompt, "")

        # Save detailed schema to file
        with open("detailed_story_schema.json", "w") as f:
            f.write(detailed_schema)

        # Step 3: Parse detailed schema and generate timeline commands
        try:
            logging.info("Starting Step 3: Parsing detailed schema and generating timeline commands")
            with open("detailed_story_schema.json", "r") as f:
                detailed_schema = json.loads(f.read())
            logging.info(f"Loaded detailed schema: {str(detailed_schema)[:100]}...")

            timeline_commands = self.video_idea_to_timeline_commands(detailed_schema)
            logging.info(f"Generated timeline commands (first 100 chars): {timeline_commands[:100]}...")

            yield {
                "message": "Timeline commands generated and saved to 'timeline_commands.json'",
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

    async def _generate_response(self, system_prompt: str, user_prompt: str):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            chat_completion_arguments = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
            }
            if openai.api_type == "azure":
                chat_completion_arguments["deployment_id"] = self.model_name

            completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

            full_response = ""
            async for chunk in completion:
                if len(chunk["choices"]) > 0:
                    if "content" in chunk["choices"][0].get("delta", {}):
                        full_response += chunk["choices"][0]["delta"]["content"]

            return full_response
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    async def generate(self, queries: list[str], context: list[str], conversation: dict = None):
        async for result in self.generate_stream(queries, context, conversation):
            pass
        return result

    def video_idea_to_timeline_commands(self, video_idea):
        logging.info("Starting video_idea_to_timeline_commands method")
        commands = {"message": {"actions": []}}
        current_time = 0.0
        total_commands = 0  # Initialize total_commands

        logging.info(f"Processing video idea with {len(video_idea['shots'])} shots")
        for shot_index, shot in enumerate(video_idea["shots"]):
            logging.info(f"Processing shot {shot_index + 1}")
            
            logging.info(f"Image layers found: {[img['layer'] for img in shot['images']]}")
            logging.info(f"Sound effects found: {[sound['layer'] for sound in shot['sound_effects']]}")

            # Image layers
            for layer in ['Background', 'Subject', 'Foreground']:
                images = [img for img in shot["images"] if img["layer"] == layer]
                for image in images:
                    commands["message"]["actions"].append({
                        "action": "insert_clip",
                        "clip": {
                            "clipPath": f"/path/to/{layer.lower()}_{shot['shot_number']}.png",
                            "length": image["duration"],
                            "track": {"Background": 1, "Subject": 2, "Foreground": 3}[layer],
                            "start": current_time,
                            "speed": 1.0
                        }
                    })
                    total_commands += 1
                    logging.info(f"Added {layer} image for shot {shot['shot_number']}")

            # Sound effects
            for i, sound in enumerate(shot["sound_effects"]):
                commands["message"]["actions"].append({
                    "action": "insert_clip",
                    "clip": {
                        "clipPath": f"/path/to/{sound['layer'].lower()}_{shot['shot_number']}.wav",
                        "length": sound["duration"],
                        "track": 4 + i,
                        "start": current_time,
                        "speed": 1.0
                    }
                })
                total_commands += 1
                logging.info(f"Added {sound['layer']} sound for shot {shot['shot_number']}")

            current_time += shot["duration"]
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