import base64
import json
from datetime import datetime
from wasabi import msg

from goldenverba.components.document import Document
from goldenverba.components.interfaces import Reader
from goldenverba.components.types import FileData

class VideoReader(Reader):
    def __init__(self):
        super().__init__()
        self.name = "VideoReader"
        self.description = "Imports video frame data from JSON files."

    def load(self, fileData: list[FileData], textValues: list[str], logging: list[dict]) -> tuple[list[Document], list[dict]]:
        documents = []
        msg.info(f"VideoReader: Processing {len(fileData)} files")

        for file in fileData:
            if file.extension == "json":
                try:
                    content = self.decode_content(file.content)
                    video_data = json.loads(content)

                    # Create a single document for the entire video
                    document = Document(
                        name=video_data[0]['original_filename'],
                        text=json.dumps(video_data),  # Store the entire JSON as text
                        type="video",
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                        metadata={
                            'total_frames': video_data[0]['total_frames'],
                            'video_length': video_data[0]['video_length'],
                            'fps': video_data[0]['fps']
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