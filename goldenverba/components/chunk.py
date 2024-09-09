from typing import List, Dict, Any

class Chunk:
    def __init__(
        self,
        text: str = "",
        doc_name: str = "",
        doc_type: str = "",
        doc_uuid: str = "",
        chunk_id: str = "",
        public_id: str = "",  
        tags: str = "",
        start: float = 0.0,
        end: float = 0.0,
        confidence: float = 0.0,
        channel: int = 0,
        speaker: float = 0.0,
        original_id: str = "",
        words: List[Dict[str, Any]] = None
    ):
        self._text = text
        self._doc_name = doc_name
        self._doc_type = doc_type
        self._doc_uuid = doc_uuid
        self._chunk_id = chunk_id
        self._tokens = 0
        self._vector = None
        self._score = 0
        self._public_id = public_id
        self._tags = tags
        self._start = start
        self._end = end
        self._confidence = confidence
        self._channel = channel
        self._speaker = speaker
        self._original_id = original_id
        self._words = words if words is not None else []
        self.meta = {}


    @property
    def text(self):
        return self._text

    @property
    def text_no_overlap(self):
        return self._text_no_overlap

    @property
    def doc_name(self):
        return self._doc_name

    @property
    def doc_type(self):
        return self._doc_type

    @property
    def doc_uuid(self):
        return self._doc_uuid

    @property
    def chunk_id(self):
        return self._chunk_id

    @property
    def tokens(self):
        return self._tokens

    @property
    def vector(self):
        return self._vector

    @property
    def score(self):
        return self._score
    

    @property
    def public_id(self):
        return self._public_id

    @property
    def tags(self):
        return self._tags
    
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def confidence(self):
        return self._confidence

    @property
    def channel(self):
        return self._channel

    @property
    def speaker(self):
        return self._speaker

    @property
    def original_id(self):
        return self._original_id

    @property
    def words(self):
        return self._words
    

    def set_public_id(self, public_id):
        self._public_id = public_id

    def set_tags(self, tags):
        self._tags = tags

    def set_uuid(self, uuid):
        self._doc_uuid = uuid

    def set_tokens(self, token):
        self._tokens = token

    def set_vector(self, vector):
        self._vector = vector

    def set_score(self, score):
        self._score = score

    def to_dict(self) -> dict:
        """Convert the Chunk object to a dictionary."""
        return {
            "text": self.text,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "doc_uuid": self.doc_uuid,
            "chunk_id": self.chunk_id,
            "tokens": self.tokens,
            "vector": self.vector,
            "score": self.score,
            "public_id": self.public_id,
            "tags": self.tags,
            "meta": self.meta,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "channel": self.channel,
            "speaker": self.speaker,
            "original_id": self.original_id,
            "words": self.words,

        }

    @classmethod
    def from_dict(cls, data: dict):
        """Construct a Chunk object from a dictionary."""
        chunk = cls(
            text=data.get("text", ""),
            doc_name=data.get("doc_name", ""),
            doc_type=data.get("doc_type", ""),
            doc_uuid=data.get("doc_uuid", ""),
            chunk_id=data.get("chunk_id", ""),
            public_id=data.get("public_id", ""),
            tags=data.get("tags", ""),
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            confidence=data.get("confidence", 0.0),
            channel=data.get("channel", 0),
            speaker=data.get("speaker", 0),
            original_id=data.get("original_id", ""),
            words=data.get("words", []),
        )
        chunk.set_tokens(data.get("tokens", 0))
        chunk.set_vector(data.get("vector", None))
        chunk.set_score(data.get("score", 0))
        return chunk
