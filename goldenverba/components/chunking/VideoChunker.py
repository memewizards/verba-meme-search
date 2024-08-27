from goldenverba.components.interfaces import Chunker
from goldenverba.components.document import Document
from goldenverba.components.chunk import Chunk
from typing import List, Dict, Any, Tuple
import json

class VideoChunker(Chunker):
    def chunk(self, documents: List[Document], logging: List[Dict[str, Any]]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        for document in documents:
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            # Parse the JSON data stored in the document's text
            video_data = json.loads(document.text)
            
            for frame_data in video_data:
                chunk = Chunk(
                    text=frame_data['description'],
                    doc_name=document.name,
                    doc_type='video_frame',
                    chunk_id=frame_data['exposed_frames'][0]
                )
                
                # Add relevant metadata to the chunk
                chunk.meta = {
                    'frame_number': frame_data['exposed_frames'][0],
                    'timestamp': frame_data['length'],
                    'total_frames': frame_data['total_frames'],
                    'video_length': frame_data['video_length'],
                    'fps': frame_data['fps']
                }
                
                document.chunks.append(chunk)
        
        return documents, logging