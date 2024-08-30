import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
import logging

load_dotenv()

# Temporarily set clip_path to the Downloads folder. Later this will be dynamically set based on the user's project.
path_to_clip = "/home/vboxuser/Downloads/"

class Clip(BaseModel):
    length: float = Field(ge=0.2, description="Length of the clip in seconds")
    track: int = Field(ge=1, description="Track number for the clip")
    start: float = Field(ge=0.0, description="Start position of the clip in seconds")
    clipPath: str = Field(description="Path to the clip file")
    speed: float = Field(default=1.0, ge=0.1, le=10.0, description="Playback speed of the clip")

class InsertClipAction(BaseModel):
    action: Literal["insert_clip"]
    clip: Clip

class CutClipAction(BaseModel):
    action: Literal["cut_clip"]
    clipPath: str = Field(description="Path to the clip to be cut")
    cutTime: float = Field(ge=0.0, description="Time at which to cut the clip")

class RemoveClipAction(BaseModel):
    action: Literal["remove_clip"]
    clipId: int = Field(description="ID of the clip to be removed")

class SearchAudioContentAction(BaseModel):
    action: Literal["search_audio_content"]
    query: str = Field(description="Search query for finding specific audio content")
    clipPath: Optional[str] = Field(default=None, description="Path to the specific clip to search within, if applicable")

class SearchClipAction(BaseModel):
    action: Literal["search_clip"]
    query: str = Field(description="Search query for finding clips in the project bin")

EditingAction = InsertClipAction | CutClipAction | RemoveClipAction | SearchAudioContentAction | SearchClipAction

class VideoEditingInstructions(BaseModel):
    actions: List[EditingAction]

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

            system_prompt = f"""
            You are an AI video editing assistant. You will be provided with a description of desired video edits.
            Your goal is to respond with structured video editing instructions, including a list of editing actions
            to be performed on the timeline. Actions can include adding clips, splitting clips, deleting clips,
            overlaying clips, replacing audio, and searching for clips.

            The clipPath should always start with "{path_to_clip}".
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
            
            # Extract only the actions array and format it correctly
            actions_dict = {"actions": [action.model_dump() for action in parsed_message.actions]}
            
            logging.info(f"Formatted Video Editing Instructions: {actions_dict}")

            return json.dumps(actions_dict)

        except Exception as e:
            logging.error(f"Error in VideoEditingGenerator: {str(e)}")
            return json.dumps({"error": f"Error generating video editing instructions: {str(e)}"})