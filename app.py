from datetime import datetime
import os
import tempfile
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

# Configure the Streamlit UI
st.set_page_config(
    page_title="DocuSense: Agentic Document Auditor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuSense")
st.subheader("Agentic RAG System for Document Auditing & Verification")

# Initialize Chat & Audit Session State
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []  # Stores dicts of query, answer, chunks, time

# Sidebar for Setup & Report Actions
with st.sidebar:
  st.header("⚙️ Configuration")
  api_key = st.text_input(
      "Gemini API Key",
      type="password",
      help="Get your key from aistudio.google.com",
  )
  uploaded_file = st.file_uploader(
      "Upload Document (PDF)", type=["pdf"], help="Upload any policy or report"
  )

  if uploaded_file and api_key:
    st.success("File uploaded & API key provided!")

  st.divider()
  st.header("📑 Audit Report")

  # Generate audit report text if history exists
  if st.session_state.chat_history:
    report_text = f"# DocuSense Compliance & Audit Report\n"
    report_text += (
        f"**Document Name:**"
        f" {uploaded_file.name if uploaded_file else 'N/A'}\n"
    )
    report_text += (
        f"**Generated On:**"
        f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    report_text += "---\n\n"

    for idx, entry in enumerate(st.session_state.chat_history, 1):
      report_text += f"### Audit Query {idx}: {entry['question']}\n"
      report_text += f"**Timestamp:** {entry['timestamp']}\n\n"
      report_text += f"**Finding:**\n{entry['answer']}\n\n"
      report_text += "**Source Grounding Evidence:**\n"
      for chunk in entry["sources"]:
        report_text += f"> {chunk}\n\n"
      report_text += "---\n\n"

    # Download Button
    st.download_button(
        label="📥 Download Audit Report (.md)",
        data=report_text,
        file_name=f"DocuSense_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

    if st.button("🗑️ Clear Chat & Audit Session"):
      st.session_state.chat_history = []
      st.rerun()
  else:
    st.caption("Perform an audit query to generate a downloadable report.")


# Cache the processed document so it doesn't reload on every interaction
@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes, file_name):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(file_bytes)
    tmp_path = tmp_file.name

  # 1. Load document
  loader = PyPDFLoader(tmp_path)
  documents = loader.load()

  # 2. Chunk document into semantic segments
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=900, chunk_overlap=120
  )
  split_docs = text_splitter.split_documents(documents)

  # 3. Generate embeddings and index in vector store
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = Chroma.from_documents(
      documents=split_docs, embedding=embeddings
  )

  # Cleanup temporary file
  os.remove(tmp_path)
  return vector_store


if api_key and uploaded_file:
  with st.spinner("Indexing document semantics into Vector DB..."):
    vector_store = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

  # Setup Gemini
  llm = ChatGoogleGenerativeAI(
      model="gemini-3.8-flash", google_api_key=api_key, temperature=0.1
  )

  # Strict Anti-Hallucination & Citation Prompt
  system_prompt = """You are DocuSense, an objective, rigorous document compliance auditor.
Your job is to answer the user's question using EXCLUSIVELY the provided context paragraphs.

RULES:
1. Base your answer STRICTLY on the facts stated in the context snippets below.
2. For EVERY claim or figure you state, cite the exact page number(s) in brackets, e.g., [Page X].
3. If the context does not contain the answer, reply: "This information is not stated in the provided document." Do NOT fabricate or assume.

Context:
{context}

Question:
{question}
"""
  prompt = ChatPromptTemplate.from_template(system_prompt)

  # Display Chat History on Screen
  for entry in st.session_state.chat_history:
    with st.chat_message("user"):
      st.write(entry["question"])
    with st.chat_message("assistant"):
      st.markdown(entry["answer"])
      with st.expander("🔍 View Retrieved Source Chunks"):
        for chunk in entry["sources"]:
          st.info(chunk)

  # Input Box for New Questions
  user_question = st.chat_input("Ask any question regarding your document...")

  if user_question:
    # 1. Display user query immediately
    with st.chat_message("user"):
      st.write(user_question)

    with st.spinner("Auditing document chunks..."):
      # 2. Retrieve relevant semantic chunks
      relevant_chunks = retriever.invoke(user_question)

      formatted_context = "\n\n".join([
          f"[Page {chunk.metadata.get('page', 0) + 1}]: {chunk.page_content}"
          for chunk in relevant_chunks
      ])

      # 3. Run LLM audit synthesis
      response = llm.invoke(
          prompt.format(context=formatted_context, question=user_question)
      )

      # 4. Clean display output
      if isinstance(response.content, str):
        answer_text = response.content
      elif isinstance(response.content, list) and len(response.content) > 0:
        item = response.content[0]
        answer_text = (
            item.get("text", "") if isinstance(item, dict) else str(item)
        )
      else:
        answer_text = str(response.content)

      # Format source chunks for display and report
      source_snippets = [
          f"Page {chunk.metadata.get('page', 0) + 1}: {chunk.page_content}"
          for chunk in relevant_chunks
      ]

      # 5. Append to session state
      st.session_state.chat_history.append({
          "question": user_question,
          "answer": answer_text,
          "sources": source_snippets,
          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      })

    # 6. Display assistant response
    with st.chat_message("assistant"):
      st.markdown(answer_text)
      with st.expander("🔍 View Retrieved Source Chunks"):
        for snippet in source_snippets:
          st.info(snippet)

    # Rerun so sidebar report button refreshes with latest query
    st.rerun()

else:
  st.info("👈 Enter your Gemini API key and upload a PDF in the sidebar to begin.")