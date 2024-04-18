# OPT_chatbot

## Project Overview

OPT Bot is a conversational chatbot specifically designed to provide assistance and answer inquiries related to the F1 Optional Practical Training (OPT) visa process. It serves as a helpful resource for users seeking to understand the intricacies of the F1 OPT visa without the need to search through extensive websites or documentation.

### Key Features
1. **Visa Inquiry**: Users can ask OPT ChatBot questions about various aspects of the F1 OPT visa process, including eligibility criteria, application procedures, documentation requirements, and post-graduation options.
2. **Contextual Understanding**: OPT ChatBot maintains context throughout the conversation, allowing users to ask follow-up questions or provide additional information for more tailored responses.
3. **Conversational Interaction**: OPT ChatBot engages users in a natural conversation, providing clear and concise explanations to help users grasp the complexities of the F1 OPT visa process.
4. **Information Retrieval**: OPT ChatBot retrieves relevant information from its database of knowledge about the F1 OPT visa, ensuring that users receive accurate and up-to-date answers to their inquiries.
5. **User-Friendly Interface**: OPT ChatBot offers a simple and intuitive chat-like interface, making it easy for users to interact with the bot and gain a better understanding of the F1 OPT visa process.



## Tools and Libraries Used

- **Python**: Programming language used for development.
- **Streamlit**: Framework for building interactive web applications with Python.
- **dotenv**: Library for loading environment variables from a `.env` file.
- **langchain**: Library for text preprocessing, embedding, and retrieval.
- **Pinecone**: Storing and querying vector indexes efficiently.
- **OpenAI**: API for accessing NLP models for text embeddings and conversational AI.


## Architecture Overview

1. **Document Preprocessing**: Text documents are loaded from a directory and split into smaller chunks for efficient processing.
2. **Vector Database Initialization**: Text chunks are embedded using the OpenAI Embeddings model, and the embeddings are stored in a Pinecone vector index.
3. **Contextual Retrieval Chain Creation**: A retrieval chain is created to retrieve relevant information based on the ongoing conversation, utilizing a combination of language models and prompts.
4. **Conversational Retrieval Chain Setup**: A conversational retrieval chain is established, allowing the bot to provide responses to user queries by retrieving information from the vector database while considering the context of the conversation.
5. **User Interface**: Streamlit provides a user-friendly interface for users to interact with the bot, displaying messages exchanged between the user and the bot in a chat-like format.

## Deployment

F1Bot is deployed on Streamlit Cloud, providing users with easy access to the conversational chatbot directly through a web browser. To access the deployed version of OPT ChatBot, simply visit the Streamlit Cloud URL where the application is hosted. Users can then interact with the chatbot interface, typing messages in the input box provided and receiving responses in real-time.

URL - 
---




