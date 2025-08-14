# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate

# load environment variables
load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

###############################   INITIALIZE CHAT MODEL   #######################################################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

###############################   PROMPT TEMPLATE (No Sources, With Chat History) ###########################################################################################

prompt = PromptTemplate.from_template("""                                
You are a helpful assistant.

Here is the conversation so far:
{chat_history}

User question:
{question}

Relevant information from the knowledge base:
{context}

Answer the user's question using the provided information and conversation history. 
If you cannot find the answer in the information, reply with: "I don't know".
Keep your answer concise and to the point.
""")

###############################   SINGLE RETRIEVAL FUNCTION  ###################################################################################################

def retrieve_once(query: str):
    """Retrieve relevant docs and return merged content."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return " ".join([doc.page_content for doc in retrieved_docs])

###############################   CONVERT CHAT HISTORY TO STRING ###############################################################################################

def format_chat_history(messages):
    """Convert list of HumanMessage/AIMessage into readable text for the prompt."""
    history_str = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"
    return history_str.strip()

###############################   STREAMLIT APP ################################################################################################################

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Input
user_question = st.chat_input("Ask me something...")

if user_question:
    # Show user question
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # --- SINGLE RETRIEVAL ---
    context = retrieve_once(user_question)

    # Format chat history for prompt
    history_text = format_chat_history(st.session_state.messages)

    # Format prompt
    formatted_prompt = prompt.format(
        chat_history=history_text,
        question=user_question,
        context=context
    )

    # Call LLM once
    response = llm.invoke(formatted_prompt)

    # Show assistant response
    final_answer = response.content.strip()
    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.messages.append(AIMessage(final_answer))
