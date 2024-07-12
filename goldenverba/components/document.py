from goldenverba.components.chunk import Chunk
from typing import Optional, List
from typing import Optional, List, Dict, Any

class Document:
    def __init__(
        self,
        text: str = "",
        type: str = "",
        name: str = "",
        path: str = "",
        link: str = "",
        timestamp: str = "",
        reader: str = "",
        tags: Optional[List[str]] = None,
        example_images: Optional[List[str]] = None,
        template_images: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        views: str = "",
        comments: str = "",
        status: str = "",
        year: str = "",
        origin: str = ""
    ):
        self._text = text
        self._type = type
        self._name = name
        self._path = path
        self._link = link
        self._timestamp = timestamp
        self._reader = reader
        self._tags = tags if tags is not None else []
        self._example_images = example_images if example_images is not None else []
        self._template_images = template_images if template_images is not None else []
        self._meta: Dict[str, Any] = meta if meta is not None else {}
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self._views = views
        self._comments = comments
        self._status = status
        self._year = year
        self._origin = origin
        self.chunks: list[Chunk] = []

    @property
    def text(self):
        return self._text

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def link(self):
        return self._link

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def reader(self):
        return self._reader

    @property
    def meta(self):
        return self._meta

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def tags(self):
        return self._tags

    @property
    def example_images(self):
        return self._example_images

    @property
    def template_images(self):
        return self._template_images
    
    @property
    def views(self):
        return self._views

    @property
    def comments(self):
        return self._comments

    @property
    def status(self):
        return self._status

    @property
    def year(self):
        return self._year

    @property
    def origin(self):
        return self._origin

    @staticmethod
    def to_json(document) -> dict:
        """Convert the Document object to a JSON dict."""
        doc_dict = {
            "text": document.text,
            "type": document.type,
            "name": document.name,
            "path": document.path,
            "link": document.link,
            "timestamp": document.timestamp,
            "reader": document.reader,
            "metadata": document.metadata,
            "tags": document.tags,
            "example_images": document.example_images,
            "template_images": document.template_images,
            "views": document.views,
            "comments": document.comments,
            "status": document.status,
            "year": document.year,
            "origin": document.origin,
            "chunks": [chunk.to_dict() for chunk in document.chunks],
        }
        return doc_dict

    @staticmethod
    def from_json(doc_dict: dict):
        """Convert a JSON dict to a Document object."""
        document = Document(
            text=doc_dict.get("text", ""),
            type=doc_dict.get("type", ""),
            name=doc_dict.get("name", ""),
            path=doc_dict.get("path", ""),
            link=doc_dict.get("link", ""),
            timestamp=doc_dict.get("timestamp", ""),
            reader=doc_dict.get("reader", ""),
            tags=doc_dict.get("tags", []),
            example_images=doc_dict.get("example_images", []),
            template_images=doc_dict.get("template_images", []),
            views=doc_dict.get("views", ""),
            comments=doc_dict.get("comments", ""),
            status=doc_dict.get("status", ""),
            year=doc_dict.get("year", ""),
            origin=doc_dict.get("origin", ""),
            meta=doc_dict.get("meta", {}),
            metadata=doc_dict.get("metadata", {})
        )
        document.chunks = [
            Chunk.from_dict(chunk_data) for chunk_data in doc_dict.get("chunks", [])
        ]
        return document