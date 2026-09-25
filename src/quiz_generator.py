import json
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag_engine import RAGEngine

class QuizGenerator:
    def __init__(self, rag_engine: RAGEngine):
        self.llm = rag_engine.llm
        self.output_parser = StrOutputParser()

    def generate_quiz(self, chunks: List[Dict[str, Any]], num_questions: int = 3, difficulty: str = "Medium") -> List[Dict[str, Any]]:
        combined_text = "\n\n".join([c["text"] for c in chunks[:10]])

        prompt = PromptTemplate.from_template(
            """You are an expert educator. Create {num_questions} practice questions based on the following text.
Difficulty level: {difficulty}

Text Context:
{text}

Return strictly a valid JSON array of objects with the following schema:
[
  {{
    "id": 1,
    "type": "mcq",
    "question": "Question text here",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_option": "A) Option 1",
    "explanation": "Why this option is correct."
  }},
  {{
    "id": 2,
    "type": "short_answer",
    "question": "Conceptual question text here",
    "ideal_answer": "Key points for a complete answer",
    "explanation": "Core concept explanation."
  }}
]

JSON Output:"""
        )

        chain = prompt | self.llm | self.output_parser
        raw_response = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "text": combined_text[:3000]
        })

        try:
            cleaned_json = raw_response.strip().strip("`json").strip("`")
            return json.loads(cleaned_json)
        except Exception:
            return [{
                "id": 1,
                "type": "short_answer",
                "question": "What is the primary concept discussed in the document?",
                "ideal_answer": "Refer to the document summary.",
                "explanation": "Extracted from document overview."
            }]

    def evaluate_user_answer(self, question: str, ideal_answer: str, user_answer: str) -> Dict[str, Any]:
        prompt = PromptTemplate.from_template(
            """Grade the user's answer to the following question.

Question: {question}
Ideal Answer: {ideal_answer}
User's Answer: {user_answer}

Provide feedback in JSON format:
{{
  "score_percent": 85,
  "status": "Correct / Partially Correct / Needs Improvement",
  "feedback": "Detailed constructive feedback on what was good and what was missed."
}}

JSON Output:"""
        )

        chain = prompt | self.llm | self.output_parser
        raw_response = chain.invoke({
            "question": question,
            "ideal_answer": ideal_answer,
            "user_answer": user_answer
        })

        try:
            cleaned = raw_response.strip().strip("`json").strip("`")
            return json.loads(cleaned)
        except Exception:
            return {
                "score_percent": 70,
                "status": "Evaluated",
                "feedback": raw_response
            }
