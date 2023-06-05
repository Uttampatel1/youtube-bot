from flask import Flask, render_template, request
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


app = Flask(__name__)

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

    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY ,model_name="gpt-3.5-turbo", temperature=0.2)

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



@app.route('/', methods=['GET', 'POST'])
def home():
    global db
    if request.method == 'POST':
        video_url = request.form['youtube_url']
        db = create_db_from_youtube_video_url(video_url)
        return render_template('index.html', db_created=True)
    return render_template('index.html', db_created=False)

@app.route('/ask', methods=['POST'])
def ask_question():
    if 'db' not in globals():
        return render_template('index.html', db_created=False, error=True)
    question = request.form['question']
    response, docs = get_response_from_query(db, question)
    return render_template('response.html', response=response, docs=docs)



if __name__ == '__main__':
    app.run(debug=True)
