import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from pydantic import BaseModel, Field
from typing import List, Dict
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SearchResult(BaseModel):
    word: str = Field(description="The word or phrase found")
    start_time: float = Field(description="Start time of the utterance containing the word")
    end_time: float = Field(description="End time of the utterance containing the word")
    speaker: int = Field(description="Speaker ID")
    utterance: str = Field(description="Full utterance containing the word")
    utterance_id: str = Field(description="Exact utterance text for identification")

class TranscriptionSearchResults(BaseModel):
    results: List[SearchResult]

class TranscriptionSearchGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.name = "TranscriptionSearchGenerator"
        self.description = "Generates search results for words or phrases in transcription data"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = False
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-0613")
        self.context_window = 10000

    async def generate_stream(self, queries: List[str], context: List[str], conversation: Dict = None, chunk_info: List[Dict] = None):
        logger.info(f"TranscriptionSearchGenerator.generate_stream called with chunk_info: {json.dumps(chunk_info, indent=2)}")
        try:
            result = await self.generate(queries, context, conversation, chunk_info)
            yield {
                "message": result,
                "finish_reason": "stop"
            }
        except Exception as e:
            logger.error(f"Error in TranscriptionSearchGenerator.generate_stream: {str(e)}")
            yield {
                "message": json.dumps({"error": str(e)}),
                "finish_reason": "error"
            }

    async def generate(self, queries: list[str], context: list[str], conversation: dict = None, chunk_info: list = None) -> str:
        logger.info("TranscriptionSearchGenerator.generate called")
        logger.info(f"Queries: {queries}")
        logger.info(f"Context (first 100 chars): {context[:100] if context else 'None'}")
        logger.info(f"Chunk info received: {json.dumps(chunk_info, indent=2)}")

        utterance_timing = {}
        for chunk in chunk_info:
            utterance_timing[chunk['transcript']] = {
                'start_time': chunk['start'],
                'end_time': chunk['end'],
                'speaker': chunk['speaker']
            }

        if not context or len(context) == 0:
            error_message = "No context provided for transcription search."
            logger.error(error_message)
            return json.dumps({"error": error_message})
        
        try:
            import openai
            logger.info(f"TranscriptionSearchGenerator.generate called with queries: {queries}")
            
            logger.info("Starting to process chunks:")
            processed_context = []
            for i, chunk in enumerate(context):
                logger.info(f"Processing chunk {i}: {chunk}")
                if isinstance(chunk, dict):
                    # If chunk is already a dictionary, use it directly
                    processed_context.append(chunk)
                elif isinstance(chunk, str):
                    # If chunk is a string, use it as content with default times
                    processed_context.append({
                        "content": chunk,
                        "start_time": 0,
                        "end_time": 0
                    })
                else:
                    logger.warning(f"Unexpected chunk type: {type(chunk)}")
                    continue

                # If chunk_info is available, update the processed chunk with additional information
                if chunk_info and i < len(chunk_info):
                    chunk_meta = chunk_info[i]
                    processed_context[-1].update({
                        "start_time": float(chunk_meta.get("start_time", 0)),
                        "end_time": float(chunk_meta.get("end_time", 0)),
                        "speaker": int(chunk_meta.get("speaker", 0)),
                        "confidence": float(chunk_meta.get("confidence", 1.0))
                    })
            
            logger.info("Finished processing chunks")
            
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
            You are an AI transcription search assistant. You will be provided with a search query and transcription data.
            Your goal is to find all occurrences the specified words or phrases in the transcription utterances and respond with
            structured search results, including the phrase found, start and end times, speaker ID, and the full utterance
            containing the phrase. A chunk's start and end times are exclusive to the chunk utterance and should not mixed with other chunk utterances.
            """

            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Search query: {' '.join(str(q) for q in queries)}"}
            ]

            transcription_data = json.dumps(processed_context)
            messages.append({"role": "user", "content": f"Transcription data: {transcription_data}"})
            logging.info(f"Transcription data sent to OpenAI: {transcription_data}")

            chat_completion_arguments = {
                "model": self.model_name,
                "messages": messages,
                "functions": [{
                    "name": "generate_transcription_search_results",
                    "description": "Generate search results from transcription data",
                    "parameters": TranscriptionSearchResults.model_json_schema()
                }],
                "function_call": {"name": "generate_transcription_search_results"}
            }

            if openai.api_type == "azure":
                chat_completion_arguments["deployment_id"] = self.model_name

            completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)

            function_response = completion.choices[0].message.function_call.arguments
            parsed_results = TranscriptionSearchResults.model_validate_json(function_response)

            # Post-process results to include correct timing information
            for result in parsed_results.results:
                if result.utterance in utterance_timing:
                    timing = utterance_timing[result.utterance]
                    result.start_time = timing['start_time']
                    result.end_time = timing['end_time']
                    result.speaker = timing['speaker']

            logging.info(f"Transcription Search Results: {parsed_results}")

            return json.dumps(parsed_results.model_dump())

        except Exception as e:
            error_message = f"Error in TranscriptionSearchGenerator: {str(e)}"
            logging.error(error_message)
            return json.dumps({"error": error_message})