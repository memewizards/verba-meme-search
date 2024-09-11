import json
import logging
from typing import List, Dict
from pydantic import BaseModel, Field
from goldenverba.components.interfaces import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ChunkResult(BaseModel):
    content: str = Field(description="The content of the chunk")
    start_time: float = Field(description="Start time of the chunk")
    end_time: float = Field(description="End time of the chunk")
    speaker: int = Field(description="Speaker ID")

class TranscriptionFinder(Generator):
    def __init__(self):
        super().__init__()
        self.name = "TranscriptionFinder"
        self.description = "Returns processed chunks from the transcription data"
        self.streamable = False

    async def generate(self, queries: list[str], context: list[str], conversation: dict = None, chunk_info: list = None) -> str:
        logging.info(f"TranscriptionFinder.generate called")
        logging.info(f"Context received: {len(context)} chunks")
        logging.info(f"Chunk meta received: {len(chunk_info) if chunk_info else None} items")

        if not context or len(context) == 0:
            error_message = "No context provided for transcription finder."
            logging.error(error_message)
            return json.dumps({"error": error_message})

        try:
            processed_chunks = self.process_chunks(context, chunk_info)
            return json.dumps({"chunks": [chunk.model_dump() for chunk in processed_chunks]})
        except Exception as e:
            error_message = f"Error in TranscriptionFinder: {str(e)}"
            logging.error(error_message)
            return json.dumps({"error": error_message})

    def process_chunks(self, context: list[str], chunk_info: list) -> List[ChunkResult]:
        processed_chunks = []
        for i, chunk in enumerate(context):
            if chunk_info and i < len(chunk_info):
                chunk_meta = chunk_info[i]
                processed_chunks.append(ChunkResult(
                    content=chunk,
                    start_time=chunk_meta.get("start_time", 0),
                    end_time=chunk_meta.get("end_time", 0),
                    speaker=chunk_meta.get("speaker", 0)
                ))
            else:
                processed_chunks.append(ChunkResult(
                    content=chunk,
                    start_time=0,
                    end_time=0,
                    speaker=0
                ))
        return processed_chunks

    async def generate_stream(self, queries: list[str], context: list[str], conversation: dict = None, chunk_info: list = None):
        logging.info(f"TranscriptionFinder.generate_stream called")
        result = await self.generate(queries, context, conversation, chunk_info)
        yield {
            "message": result,
            "finish_reason": "stop"
        }