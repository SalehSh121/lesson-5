from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

class OutputFormatter:
    """Component to format retrieve context and wrap parsers."""

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """Formats retrieved documents with metadata and page content for the prompt context."""
        formatted_chunks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "unknown")
            department = doc.metadata.get("department", "unknown")
            access_level = doc.metadata.get("access_level", "unknown")

            formatted_chunks.append(
                f"[Chunk {i}]\n"
                f"Source: {source}\n"
                f"Chunk Index: {chunk_index}\n"
                f"Department: {department}\n"
                f"Access Level: {access_level}\n\n"
                f"Content:\n{doc.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    @staticmethod
    def get_parser() -> StrOutputParser:
        """Returns a string output parser."""
        return StrOutputParser()
