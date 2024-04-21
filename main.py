# Import required libraries
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
import streamlit as st 

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Function for preprocessing documents
def doc_preprocessing():
    # Load text documents from a directory
    loader = DirectoryLoader('input_files/', 
                             glob="./*.txt",                          #only txt files
                             loader_cls=TextLoader,
                             show_progress=True)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                 chunk_overlap=50)
    split_data = text_splitter.split_documents(documents)
    return split_data

# Function to initialize the vector database
def vector_db():
    pc = Pinecone(pineconeapi_key=PINECONE_API_KEY)
    index_name = 'f1textindex'
    model_name = 'text-embedding-ada-002'  
    embeddings = OpenAIEmbeddings(model=model_name, 
                                        openai_api_key=OPENAI_API_KEY)
    indexes = None
    try:
        # Try to retrieve vectors from existing index
        index = pc.Index(index_name)
        describe_stats = index.describe_index_stats()
        total_vector_count = describe_stats['total_vector_count']
        print(total_vector_count)
        if total_vector_count == 0:
            raise Exception("Total Vector Count is 0")
        # If vectors exists, load it
        indexes = PineconeVectorStore.from_existing_index(index_name, embeddings)
    except Exception:
        # If index retrieval fails or total vector count is 0, create vector
        split_data = doc_preprocessing() 
        indexes = PineconeVectorStore.from_documents(split_data, 
                                                        embeddings, 
                                                        index_name="f1textindex")
        print(indexes)
    return indexes

# Function to create retriever chain based on chat history
def get_context_retriever_chain(result):
    llm = ChatOpenAI()
    retriever = result.as_retriever()
    # Define a prompt template for generating search queries
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain    

# Function to create conversational retrieval chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    # Define a prompt template for answering user's questions
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response to user input
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with OPT ChatBot")

# sidebar
st.header("Settings")
   
# session state
if "chat_history" not in st.session_state:
        # Initialize chat history with a greeting message
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a F1 OPT Chatbot. How can I help you with today?"),
        ]
if "vector_store" not in st.session_state:
        # Initialize vector store
        st.session_state.vector_store = vector_db()

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
        # Get response to user input
        response = get_response(user_query)
        # Append user input and bot response to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
         with st.chat_message("AI"):
                st.write(message.content)
    elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
