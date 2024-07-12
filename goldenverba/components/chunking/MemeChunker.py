from goldenverba.components.interfaces import Chunker
from goldenverba.components.document import Document
print(Document.__init__)  # See the signature of the __init__ method to confirm it's the right one
from typing import List, Dict, Any, Tuple
import random
from goldenverba.components.document import Document
from goldenverba.components.chunk import Chunk

class MemeChunker(Chunker):
    def chunk(self, documents: List[Document], logging: List[Dict[str, Any]]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        for document in documents:
            chunk = Chunk(
                text=document.text,
                doc_name=document.name,
                doc_type=document.type,
                chunk_id=random.randint(1, 1000000),
                public_id=document.template_images[0] if document.template_images else '',
                tags=", ".join(document.tags)
            )
            
            # Add new properties to the chunk's metadata
            chunk.meta = {
                **document.metadata,
                "views": document.views,
                "comments": document.comments,
                "status": document.status,
                "year": document.year,
                "origin": document.origin,
                "example_images": document.example_images,
                "template_images": document.template_images
            }
            
            document.chunks = [chunk]
        
        return documents, logging