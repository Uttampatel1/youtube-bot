import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

OPENAI_API_KEY = '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'
# OPENAI_API_KEY = "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"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model='text-embedding-ada-002')


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 800 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY , temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def main():
    st.title("YouTube Video Q&A App")
    global OPENAI_API_KEY
    OPENAI_API_KEY = st.text_input("Enter Your OpenAI API key:")

    video_url = st.text_input("Enter YouTube Video URL")
    
    
    if st.button("Train Model"):
        if OPENAI_API_KEY:
            db = create_db_from_youtube_video_url(video_url)
            st.write('Train Model is successfully!')
        
    
    query = st.text_input("Ask a question")
    
    if st.button("Get Answer"):
        if OPENAI_API_KEY:
            response, docs = get_response_from_query(db, query)
            st.text_area("Answer", value=textwrap.fill(response, width=50))


if __name__ == "__main__":
    main()
