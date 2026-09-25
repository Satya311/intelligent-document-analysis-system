from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import config

class RAGEngine:
    def __init__(self):
        self.llm = self._init_llm()
        self.output_parser = StrOutputParser()

    def _init_llm(self):
        if config.LLM_PROVIDER == "google" and config.GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(
                model=config.DEFAULT_MODEL_NAME,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.3
            )
        elif config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
            return ChatOpenAI(
                model=config.DEFAULT_MODEL_NAME,
                openai_api_key=config.OPENAI_API_KEY,
                temperature=0.3
            )
        else:
            raise ValueError("Valid API Key (Google or OpenAI) is required for RAG Engine.")

    def answer_question(self, query: str, retrieved_docs: List[Any]) -> Dict[str, Any]:
        context_str = ""
        citations = []

        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata
            cid = f"[Source {i+1} | Page {meta.get('page_number')} | {meta.get('content_type')}]"
            context_str += f"{cid}\n{doc.page_content}\n\n"
            citations.append({
                "source_id": i + 1,
                "page": meta.get("page_number"),
                "content_type": meta.get("content_type"),
                "topic": meta.get("topic_name"),
                "snippet": doc.page_content[:200] + "..."
            })

        prompt_template = PromptTemplate.from_template(
            """You are an Intelligent Learning Assistant helping a reader grasp document concepts quickly.

Context from Document:
{context}

User Question: {question}

Instructions:
1. Provide a clear, structured, easy-to-understand answer based on the context.
2. Highlight key terms in **bold**.
3. Cite sources inline using [Source X].
4. If context is insufficient, state clearly what is missing.

Answer:"""
        )

        chain = prompt_template | self.llm | self.output_parser
        response_text = chain.invoke({"context": context_str, "question": query})

        return {"answer": response_text, "citations": citations}
