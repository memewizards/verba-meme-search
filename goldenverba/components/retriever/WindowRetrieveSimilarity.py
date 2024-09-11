from weaviate import Client
from weaviate.gql.get import HybridFusion

from goldenverba.components.chunk import Chunk
from goldenverba.components.interfaces import Embedder, Retriever
import time

class WindowRetrieveSimilarity(Retriever):
    """
    WindowRetriever that retrieves chunks and their surrounding context depending on the window size.
    """

    def __init__(self):
        super().__init__()
        self.description = "similary search. allegedly faster"
        self.name = "WindowRetriever"



    def retrieve(
        self,
        queries: list[str],
        client: Client,
        embedder: Embedder,
        doc_type: str = None,
        limit: int = 5
    ) -> tuple[list[Chunk], str]:
        chunk_class = embedder.get_chunk_class()
        chunks = []

        total_query_time = 0

        for query in queries:
            start_time = time.time()

            query_results = (
                client.query.get(
                    class_name=chunk_class,
                    properties=[
                        "text",
                        "doc_name",
                        "chunk_id",
                        "doc_uuid",
                        "doc_type",
                    ],
                )
                .with_additional(properties=["score"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
            )

            if doc_type:
                query_results = query_results.with_where({
                    "path": ["doc_type"],
                    "operator": "Equal",
                    "valueString": doc_type
                })

            results = query_results.do()

            end_time = time.time()
            query_time = end_time - start_time
            total_query_time += query_time

            print(f"Query '{query}' took {query_time:.2f} seconds")

            print(f"Total query time for all queries: {total_query_time:.2f} seconds")


            for chunk in results["data"]["Get"][chunk_class]:
                chunk_obj = Chunk(
                    chunk["text"],
                    chunk["doc_name"],
                    chunk["doc_type"],
                    chunk["doc_uuid"],
                    chunk["chunk_id"],
                )
                chunk_obj.set_score(chunk["_additional"]["score"])
                chunks.append(chunk_obj)

        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        context = self.combine_context(sorted_chunks)

        return sorted_chunks, context

    def combine_context(self, chunks: list[Chunk]) -> str:
        context = ""
        for chunk in chunks:
            context += f"--- Document: {chunk.doc_name} ---\n\n"
            context += f"Chunk {chunk.chunk_id}\n\n{chunk.text}\n\n"
        return context
