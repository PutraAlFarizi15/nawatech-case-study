"""
import package
"""
import os
import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Configuration and API Key Loading ---
# Load environment variables from the .env file
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Nawatech FAQ Chatbot (OpenAI)", page_icon="🤖", layout="centered")

# Get the API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Constants ---
DATA_PATH = "data/FAQ_Nawa.xlsx"

# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
    """Loads FAQ data from an Excel file."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df.dropna(subset=['Question', 'Answer'], inplace=True)
        df['Question'] = df['Question'].str.strip()
        df['Answer'] = df['Answer'].str.strip()
        docs = [
            Document(
                page_content=f"Question: {row['Question']}\nAnswer: {row['Answer']}",
                # Metadata can be used to store the source or additional information
                metadata={'source': file_path, 'row_number': index}
            )
            for index, row in df.iterrows()
        ]
        return docs
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None

# Change: _embeddings is now a parameter to allow for embedding model replacement
@st.cache_resource
def create_vector_db(_docs, _embeddings):
    """Creates a FAISS vector store from documents."""
    if not _docs:
        return None
    try:
        vector_store = FAISS.from_documents(_docs, _embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector database: {e}. Make sure your OpenAI API Key is valid.")
        return None
def format_docs(docs):
    """ Combines the content of all found documents into a single string """
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Application Logic ---

st.title("🤖 Nawatech FAQ Chatbot (OpenAI Version)")
st.markdown("I am ready to answer your questions about Nawatech using technology from OpenAI.")

# 1. Initialize chat history in session_state
if "messages" not in st.session_state:
    # Create an empty list when the application is first run
    st.session_state.messages = []
    # Add a welcome message from the assistant
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! How can I help you with Nawatech?"}
    )

# 2. Display all messages from the chat history
# This loop will run every time there is an interaction to re-render the UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Validate API Key before proceeding
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in your .env file.")
else:
    faq_docs = load_data(DATA_PATH)
    if faq_docs:
        # Change: Use OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_db = create_vector_db(faq_docs, embeddings)

        if vector_db:
            # 1. Define Prompt Template 
            prompt_template = """
            You are a friendly AI assistant for Nawatech. Use the following context to answer the user's question accurately.
            If the information is not found in the context, politely say that you do not have the information. Do not try to make up an answer.

            Context: {context}
            Question: {question}

            Helpful Answer:
            """
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
                )
            
            # 2. Define LLM 
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                api_key=OPENAI_API_KEY
            )

            # 3. Create a Retriever from the Vector Store
            # This is the standard way to retrieve documents in an LCEL chain
            retriever = vector_db.as_retriever(search_kwargs={'k': 3})

            # 4. Chain everything together using LCEL
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | PROMPT
                | llm
                | StrOutputParser()
            )

            # ... (inside your chat input loop)
            if prompt := st.chat_input("Ask something about Nawatech"):
                # ... (logic to display user message)
                # Display user message in the UI
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Save user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.spinner("Thinking..."):
                    try:
                        # The way to call the LCEL chain is slightly different
                        # You no longer need to perform a similarity_search manually
                        # The retriever in the chain will do it for you.
                        answer = rag_chain.invoke(prompt)
                        
                        # If you still want to display quality based on score,
                        # you still need to do a manual search first like in your original code.
                        # However, to get the answer, invoke() is sufficient.
                        retrieved_docs_with_scores = vector_db.similarity_search_with_score(prompt, k=3)
                        if not retrieved_docs_with_scores:
                            quality = "Not Found"
                        else:
                            scores = [score for doc, score in retrieved_docs_with_scores]
                            avg_score = sum(scores) / len(scores)
                            if avg_score < 0.35:
                                quality = "High"
                            elif avg_score < 0.6:
                                quality = "Medium"
                            else:
                                quality = "Low"
                                
                        # Display assistant's answer in the UI
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            st.info(f"**Answer Quality:** {quality}", icon="✅")
                            
                        # Save assistant's answer to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        response = f"Sorry, an error occurred: {e}"
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Chatbot cannot run because the FAQ data failed to load.")