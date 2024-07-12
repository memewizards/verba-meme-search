import base64
import json
from datetime import datetime
from wasabi import msg


from datetime import datetime
from wasabi import msg
from goldenverba.components.document import Document
from goldenverba.components.interfaces import Reader
from goldenverba.components.types import FileData

class CustomMemeReader(Reader):
    def __init__(self):
        super().__init__()
        self.name = "CustomMemeReader"
        self.description = "Imports custom meme JSON files."

    print("THE CUSTOMMEME READER IS BEING CREATED")

    def load(self, fileData: list[FileData], textValues: list[str], logging: list[dict]) -> tuple[list[Document], list[dict]]:
        documents = []
        msg.info(f"CustomMemeReader: Processing {len(fileData)} files")

        for file in fileData:
            if file.extension == "json":
                try:
                    content = self.decode_content(file.content)
                    meme_data = json.loads(content)

                    document = Document(
                        name=meme_data['meme_id'],
                        text=meme_data['about'],
                        type="meme",
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                        tags=meme_data.get('tags', []),
                        example_images=meme_data.get('example_images', []),
                        template_images=meme_data.get('template_images', []),
                        metadata={
                            'views': meme_data.get('views', 0),
                            'comments': meme_data.get('comments', 0),
                            'type': meme_data.get('type', []),
                            'status': meme_data.get('status', '')
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