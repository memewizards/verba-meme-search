import logging
from goldenverba.components.interfaces import Chunker
from goldenverba.components.document import Document
from goldenverba.components.chunk import Chunk
from typing import List, Dict, Any, Tuple
import json

logger = logging.getLogger(__name__)

class TranscriptionChunker(Chunker):
    def chunk(self, documents: List[Document], logging: List[Dict[str, Any]]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        for document in documents:
            if document.type != "transcription":
                logging.append({
                    "type": "WARNING",
                    "message": f"Skipping document {document.name}: Not a transcription (Type: {document.type})"
                })
                continue

            try:
                transcription_data = json.loads(document.metadata['full_content'])
                utterances = transcription_data.get('utterances', [])
                chunk_info = document.metadata.get('chunk_info', [])
                chunks = []

                for index, chunk_data in enumerate(chunk_info):
                    chunk = Chunk(
                        text=chunk_data['transcript'],  # Only the transcript text goes here
                        doc_name=document.name,
                        doc_type=document.type,
                        chunk_id=float(index),
                    )

                    # Store all metadata separately
                    chunk.meta = {
                        'start': chunk_data['start'],
                        'end': chunk_data['end'],
                        'confidence': chunk_data['confidence'],
                        'channel': chunk_data['channel'],
                        'speaker': chunk_data['speaker'],
                        'original_id': chunk_data['original_id'],
                        'words': chunk_data['words']
                    }
                    
                    chunks.append(chunk)

                document.chunks = chunks
                document.metadata['chunks_count'] = len(chunks)
                document.metadata['text_processed'] = True

            except json.JSONDecodeError as json_err:
                error_message = (
                    f"Error processing document {document.name}: "
                    f"Invalid JSON in 'full_content' metadata. "
                    f"JSON error: {str(json_err)}. "
                    f"First 100 characters of content: '{document.metadata['full_content'][:100]}...'"
                )
                logger.error(error_message)
                logging.append({
                    "type": "ERROR",
                    "message": error_message,
                    "document_name": document.name,
                    "error_type": "JSONDecodeError",
                    "error_details": str(json_err)
                })
            except KeyError as key_err:
                error_message = (
                    f"Error processing document {document.name}: "
                    f"Missing key in document metadata: {str(key_err)}"
                )
                logger.error(error_message)
                logging.append({
                    "type": "ERROR",
                    "message": error_message,
                    "document_name": document.name,
                    "error_type": "KeyError",
                    "error_details": str(key_err)
                })
            except Exception as e:
                error_message = (
                    f"Error processing document {document.name}: {str(e)}. "
                    f"Document type: {document.type}. "
                    f"Metadata keys: {', '.join(document.metadata.keys())}"
                )
                logger.error(error_message)
                logging.append({
                    "type": "ERROR",
                    "message": error_message,
                    "document_name": document.name,
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                })

        return documents, logging 