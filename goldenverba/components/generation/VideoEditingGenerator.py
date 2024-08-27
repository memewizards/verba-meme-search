import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from pydantic import BaseModel, Field, validator
from typing import List
import logging

load_dotenv()


class Clip(BaseModel):
    length: float = Field(ge=0.2, description="Length of the clip in seconds")
    track: int = Field(ge=1, description="Track number for the clip")
    start: float = Field(ge=0.0, description="Start position of the clip in seconds")
    clipPath: str = Field(description="Path to the clip file")  # Changed from clip_path to clipPath

    @validator('length')
    def length_must_be_positive(cls, v):
        return max(v, 0.0)

    @validator('track')
    def track_must_be_positive(cls, v):
        return max(v, 1)

    @validator('start')
    def start_must_be_non_negative(cls, v):
        return max(v, 0.0)

class VideoEditingInstructions(BaseModel):
    clips: List[Clip]

class VideoEditingGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.name = "VideoEditingGenerator"
        self.description = "Generates structured video editing instructions"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = False
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
        self.context_window = 10000

    async def generate_stream(self, queries: list[str], context: list[str], conversation: dict = None):
        result = await self.generate(queries, context, conversation)
        yield {
            "message": result,
            "finish_reason": "stop"
        }

    async def generate(self, queries: list[str], context: list[str], conversation: dict = None) -> str:
        try:
            import openai
            logging.info(f"VideoEditingGenerator.generate called with queries: {queries}")

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

            system_prompt = """
            You are an AI video editing assistant. You will be provided with a description of desired video edits.
            Your goal is to respond with structured video editing instructions, including a list of clips to be added
            to the timeline. For each clip, provide the length, track number, start position, and clipPath.
            The clipPath should always start with '/home/vboxuser/Downloads/'.
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": " ".join(queries)}
            ]

            if context:
                messages.append({"role": "user", "content": f"Context: {' '.join(context)}"})

            chat_completion_arguments = {
                "model": self.model_name,
                "messages": messages,
                "functions": [{
                    "name": "generate_video_editing_instructions",
                    "description": "Generate structured video editing instructions",
                    "parameters": VideoEditingInstructions.model_json_schema()
                }],
                "function_call": {"name": "generate_video_editing_instructions"}
            }

            if openai.api_type == "azure":
                chat_completion_arguments["deployment_id"] = self.model_name

            completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

            function_response = completion.choices[0].message.function_call.arguments
            parsed_message = VideoEditingInstructions.model_validate_json(function_response)
            
            # Extract only the clips array and format it correctly
            clips_dict = {"clips": [clip.model_dump() for clip in parsed_message.clips]}
            
            logging.info(f"Formatted Video Editing Instructions: {clips_dict}")

            return json.dumps(clips_dict)

        except Exception as e:
            logging.error(f"Error in VideoEditingGenerator: {str(e)}")
            return json.dumps({"error": f"Error generating video editing instructions: {str(e)}"})

    def prepare_messages(self, queries: list[str], context: list[str], conversation: dict[str, str]) -> dict[str, str]:
        # This method is not used in this generator, but we'll keep it for consistency
        pass