import streamlit as st
import config
from src.loader import DocumentLoader
from src.ml_classifier import ConventionalMLAnalyzer
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine
from src.quiz_generator import QuizGenerator

st.set_page_config(
    page_title="Intelligent Document Analysis & Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Document Analysis & Practice System")
st.caption("Gen AI (RAG + LLM) + Conventional ML (Classification & Topic Clustering)")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Google / OpenAI API Key", type="password")
    if api_key_input:
        config.GOOGLE_API_KEY = api_key_input
        config.OPENAI_API_KEY = api_key_input

    difficulty = st.selectbox("Quiz Difficulty", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("Number of Practice Questions", 1, 5, 3)

if "enriched_chunks" not in st.session_state:
    st.session_state.enriched_chunks = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

uploaded_file = st.file_uploader("Upload a Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file and st.button("🚀 Analyze & Index Document"):
    with st.spinner("Processing document, running ML clustering, and indexing vectors..."):
        loader = DocumentLoader()
        chunks = loader.process_document(uploaded_file.getvalue(), uploaded_file.name)

        ml_analyzer = ConventionalMLAnalyzer(num_clusters=config.NUM_TOPIC_CLUSTERS)
        enriched_chunks = ml_analyzer.enrich_chunks(chunks)
        st.session_state.enriched_chunks = enriched_chunks

        vs_manager = VectorStoreManager()
        vs_manager.build_index(enriched_chunks)
        st.session_state.vector_store = vs_manager

        if config.GOOGLE_API_KEY or config.OPENAI_API_KEY:
            st.session_state.rag_engine = RAGEngine()

        st.success(f"Successfully processed {len(enriched_chunks)} document chunks across topic clusters!")

tab1, tab2, tab3 = st.tabs(["📊 ML Topic Analysis", "💬 Smart Q&A Chat (RAG)", "🎯 Practice Quiz Arena"])

with tab1:
    st.subheader("Conventional ML Structure & Topic Breakdown")
    if st.session_state.enriched_chunks:
        chunks = st.session_state.enriched_chunks
        topics = list(set([c.get("topic_name") for c in chunks]))
        st.write(f"**Identified Topic Clusters ({len(topics)}):**")
        for t in topics:
            st.markdown(f"- 🔹 {t}")

        st.write("### Chunk Breakdown & Classification")
        table_data = []
        for c in chunks[:15]:
            table_data.append({
                "Chunk ID": c["chunk_id"],
                "Page": c["page_number"],
                "Topic Cluster": c.get("topic_name"),
                "Content Type (ML Classified)": c.get("content_type"),
                "Snippet": c["text"][:120] + "..."
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("Upload and process a document to view conventional ML topic clusters.")

with tab2:
    st.subheader("Context-Aware Q&A with Citation Support")
    if not st.session_state.vector_store:
        st.warning("Please upload a document and build the index first.")
    elif not st.session_state.rag_engine:
        st.warning("Please enter your API Key in the sidebar to enable Gen AI Q&A.")
    else:
        user_query = st.text_input("Ask a question about your document:")
        if user_query and st.button("Ask Assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                retrieved_docs = st.session_state.vector_store.similarity_search(user_query, top_k=3)
                rag_res = st.session_state.rag_engine.answer_question(user_query, retrieved_docs)

                st.markdown("### Answer")
                st.write(rag_res["answer"])

                st.markdown("---")
                st.markdown("### 📌 Retrieved Sources & Citations")
                for cite in rag_res["citations"]:
                    st.markdown(f"**Source {cite['source_id']}** (Page {cite['page']} | {cite['content_type']} | {cite['topic']}):")
                    st.caption(f'"{cite["snippet"]}"')

with tab3:
    st.subheader("Interactive Practice & Assessment Arena")
    if not st.session_state.enriched_chunks or not st.session_state.rag_engine:
        st.info("Upload document and configure API key to generate practice questions.")
    else:
        if st.button("🎲 Generate Practice Quiz"):
            quiz_gen = QuizGenerator(st.session_state.rag_engine)
            st.session_state.quiz_data = quiz_gen.generate_quiz(
                st.session_state.enriched_chunks,
                num_questions=num_questions,
                difficulty=difficulty
            )

        if st.session_state.quiz_data:
            st.markdown("---")
            for q in st.session_state.quiz_data:
                st.markdown(f"#### Question {q['id']}: {q['question']}")
                if q["type"] == "mcq":
                    selected = st.radio(f"Select your answer for Q{q['id']}:", q["options"], key=f"q_{q['id']}")
                    if st.button(f"Check Q{q['id']} Answer", key=f"btn_{q['id']}"):
                        if selected == q["correct_option"]:
                            st.success("🎉 Correct!")
                        else:
                            st.error(f"Incorrect. Correct answer: {q['correct_option']}")
                        st.info(f"Explanation: {q['explanation']}")
                else:
                    user_ans = st.text_area(f"Your answer for Q{q['id']}:", key=f"txt_{q['id']}")
                    if st.button(f"Submit Q{q['id']} Answer", key=f"sub_{q['id']}"):
                        quiz_gen = QuizGenerator(st.session_state.rag_engine)
                        eval_res = quiz_gen.evaluate_user_answer(q["question"], q["ideal_answer"], user_ans)
                        st.metric("Score", f"{eval_res.get('score_percent', 0)}%")
                        st.write(f"**Status:** {eval_res.get('status')}")
                        st.write(f"**Feedback:** {eval_res.get('feedback')}")
