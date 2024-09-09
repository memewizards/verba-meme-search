import base64
import json
from datetime import datetime
from wasabi import msg

from goldenverba.components.document import Document
from goldenverba.components.interfaces import Reader
from goldenverba.components.types import FileData

class TranscriptionReader(Reader):
    def __init__(self):
        super().__init__()
        self.name = "TranscriptionReader"
        self.description = "Imports audio transcription data from JSON files."

    def load(self, fileData: list[FileData], textValues: list[str], logging: list[dict], config: dict = None) -> tuple[list[Document], list[dict]]:
        documents = []
        msg.info(f"TranscriptionReader: Processing {len(fileData)} files")

        for file in fileData:
            if file.extension == "json":
                try:
                    content = self.decode_content(file.content)
                    transcription_data = json.loads(content)
                    utterances = transcription_data.get('utterances', [])
                    
                    # Concatenate utterances to create full text
                    full_text = " ".join([utterance.get('transcript', '') for utterance in utterances])
    
                    # Create chunk_info
                    chunk_info = []
                    for index, utterance in enumerate(utterances):
                        chunk_info.append({
                            'chunk_id': str(index),
                            'transcript': utterance.get('transcript', ''),
                            'start': utterance.get('start', 0.0),
                            'end': utterance.get('end', 0.0),
                            'confidence': utterance.get('confidence', 0.0),
                            'channel': utterance.get('channel', 0),
                            'speaker': str(utterance.get('speaker', '0')),
                            'original_id': utterance.get('utterance_id', ''),
                            'words': utterance.get('words', []),
                            'duration': utterance.get('duration', 0.0),

                        })

                    document = Document(
                        name=f"Transcription_{transcription_data.get('video_id', 'unknown')}",
                        text=full_text,  # Store the concatenated full text
                        type="transcription",
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                        metadata={
                            'video_id': transcription_data.get('video_id'),
                            'sha256': transcription_data.get('sha256'),
                            'utterances_count': len(utterances),
                            'full_content': content,
                            'chunk_info': chunk_info
                        }
                    )

                    documents.append(document)
                    msg.good(f"Successfully processed: {file.filename}")
                except Exception as e:
                    msg.warn(f"Failed to load {file.filename} : {str(e)}")
                    logging.append({
                        "type": "WARNING",
                        "message": f"Failed to load {file.filename} : {str(e)}",
                    })
        return documents, logging

    def decode_content(self, content):
        try:
            return base64.b64decode(content).decode('utf-8')
        except:
            return content