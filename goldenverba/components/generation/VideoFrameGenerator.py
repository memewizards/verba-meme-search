import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from pydantic import BaseModel, Field
import logging

load_dotenv()

class VideoFrame(BaseModel):
    frame_number: int = Field(ge=0, description="Frame number")
    timestamp: str = Field(description="Timestamp of the frame")
    description: str = Field(description="Description of the frame")

class VideoFrameGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.name = "VideoFrameGenerator"
        self.description = "Generates structured video frame information for the most relevant frame"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = True
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.context_window = 10000

    async def generate_stream(self, queries: list[str], context: list[str], conversation: dict = None):
        try:
            import openai
            logging.info(f"VideoFrameGenerator.generate_stream called with queries: {queries}")

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
            You are an AI video frame analyzer. You will be provided with descriptions of video frames.
            Your goal is to identify the most relevant frame based on the given query and respond with
            structured information about that single frame, including the frame number, timestamp, and description.
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {' '.join(queries)}"}
            ]

            if context:
                messages.append({"role": "user", "content": f"Context: {' '.join(context)}"})

            chat_completion_arguments = {
                "model": self.model_name,
                "messages": messages,
                "functions": [{
                    "name": "generate_most_relevant_video_frame",
                    "description": "Generate structured information for the most relevant video frame",
                    "parameters": VideoFrame.model_json_schema()
                }],
                "function_call": {"name": "generate_most_relevant_video_frame"},
                "stream": True
            }

            if openai.api_type == "azure":
                chat_completion_arguments["deployment_id"] = self.model_name

            completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

            full_response = ""
            async for chunk in completion:
                if chunk.choices[0].delta.get("function_call"):
                    if "arguments" in chunk.choices[0].delta.function_call:
                        full_response += chunk.choices[0].delta.function_call.arguments
                        yield {
                            "message": chunk.choices[0].delta.function_call.arguments,
                            "finish_reason": chunk.choices[0].finish_reason
                        }

            parsed_message = VideoFrame.model_validate_json(full_response)
            frame_dict = parsed_message.model_dump()
            
            logging.info(f"Most Relevant Video Frame Information: {frame_dict}")

            yield {
                "message": json.dumps(frame_dict),
                "finish_reason": "stop"
            }

        except Exception as e:
            logging.error(f"Error in VideoFrameGenerator: {str(e)}")
            yield {
                "message": json.dumps({"error": f"Error generating video frame information: {str(e)}"}),
                "finish_reason": "error"
            }

    def prepare_messages(self, queries: list[str], context: list[str], conversation: dict[str, str]) -> dict[str, str]:
        # This method is not used in this generator, but we'll keep it for consistency
        pass