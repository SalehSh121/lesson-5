from langchain_google_genai import ChatGoogleGenerativeAI

class LLMService:
    """Component to manage LLM instances."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Returns the configured Chat LLM instance."""
        return self.llm
