# pip install pycryptodome, pymupdf
from glob import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import json
import os

#QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ( "GPT-3.5-16k", "GPT-4", "GPT-3.5"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    
    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []

        #  memoryの初期化
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

def get_pdf_text(uploaded_file):
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    #client = QdrantClient(path=QDRANT_PATH)

    # 以前こう書いていたところ: client = QdrantClient(path=QDRANT_PATH)
    # url, api_key は Qdrant Cloud から取得する
    client = QdrantClient(
        url=os.environ['QDRANT_CLOUD_ENDPOINT'],
        api_key=os.environ['QDRANT_CLOUD_API_KEY']
    )

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )


def build_qa_model(llm, memory):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":4}
    )

    return ConversationalRetrievalChain.from_llm(
       llm, retriever=retriever, memory=memory, verbose=True
    )


def page_pdf_upload_and_build_vector_db():
    # PDFファイルをアップロードする欄を作ります。
    uploaded_file = st.sidebar.file_uploader(
            label='Upload your PDF here😇',
            type='pdf'
        )
    if not uploaded_file:
            st.info("PDFファイルをアップロードしてください")
            st.stop()

    container = st.container()
    with container:
        pdf_text = get_pdf_text(uploaded_file)
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa.run(query)

    return answer, cb.total_cost


def page_ask_my_pdf():

    llm = select_model()

    # 会話のコンテキストを管理するメモリを設定する
    memory = st.session_state.memory 

    qa_chain = build_qa_model(llm, memory)

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = ask(qa_chain, user_input)
            st.session_state.costs.append(cost)

        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=answer))

        messages = st.session_state.get('messages', [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")



# メッセージオブジェクトを辞書に変換する関数
def message_to_dict(message):
    return {
        'type': type(message).__name__,
        'content': message.content,
    }


def main():
    init_page()
    init_messages()

    st.title("PDFと対話しよう！")

    page_pdf_upload_and_build_vector_db()

    page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


    messages_as_dicts = [message_to_dict(message) for message in st.session_state.messages]

    # ensure_ascii=Falseを設定
    messages_str = json.dumps(messages_as_dicts, ensure_ascii=False)

    st.download_button(
        label="Download messages",
        data=messages_str.encode(),  # bytesに変換
        file_name='messages.json',
        mime='application/json',
    )

if __name__ == '__main__':
    main()
