from PIL import Image
import pytesseract
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tempfile, os, chromadb, pandas as pd, pdfplumber
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredCSVLoader
from operator import itemgetter
import chromadb
# customize the intial app landing page
st.set_page_config(page_title='file QA Chatbot', page_icon="🤖")
st.title('Multi-Document AI QA RAG Chatbot🤖')
uploaded_files=st.sidebar.file_uploader(label="Upload Files(PDF,TXT,DOCX,CSV,Images)",
                                        type=["pdf", "txt", "docx", "csv", "png", "jpg", "jpeg"],
                                        accept_multiple_files=True)
if not uploaded_files:
    st.info(f"Upload a file one or more documents.")
    st.stop()

#Pdfplumber loader
os.makedirs('extractimages', exist_ok=True)

def load_pdfplumber(file_path):
  documents=[]
  with pdfplumber.open(file_path) as pdf:
    for i,page in  enumerate(pdf.pages):
      text=page.extract_text()
      images=[]
      for img_idex,img in enumerate(page.images):
        x0=int(round(img['x0']))
        top=int(round(img['top']))
        x1=int(round(img['x1']))
        bottom=int(round(img['bottom']))
        cropped= page.crop((x0,top,x1,bottom),strict=False)
        image=cropped.to_image(resolution=300)
        imagepath=f'extractimages/page_{i+1}_img_{img_idex}.png'
        try:
          image.save(imagepath, format='PNG')
          images.append(imagepath)
        except SystemError as e:
          print(f'image:{img_idex+1}.error:{e}')

      documents.append(Document(page_content=text,metadata={'page':i+1,'images':images}))
  return documents

def load_image_with_ocr(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    metadata = {"source": os.path.basename(file_path)}
    return [Document(page_content=text, metadata=metadata)]


# vector store and Retriver
@st.cache_resource(ttl='1h')# This caches the result for 1 hour, so it doesn’t reprocess files every time the app reloads.
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        temp_path = os.path.join(temp_dir.name, file.name)

        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        # Load based on extension
        if ext == "pdf":
            docs.extend(load_pdfplumber(temp_path))
        elif ext == "txt":
            docs.extend(TextLoader(temp_path).load())
        elif ext == "docx":
            docs.extend(Docx2txtLoader(temp_path).load())
        elif ext == "csv":
            docs.extend(UnstructuredCSVLoader(temp_path).load())
        elif ext in ["jpg", "jpeg", "png"]:
            docs.extend(load_image_with_ocr(temp_path))
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create Vector DB
    embeddings = OpenAIEmbeddings()
    client = chromadb.PersistentClient(path="./chroma_db")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="multi_doc_collection"
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

retriever = configure_retriever(uploaded_files)

# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)
# This function formats retrieved documents before sending to LLM
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])


# LangChain Pipeline
# --------------------------
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.1, streaming=True)

qa_chain = (
    {
        "context": itemgetter("question") # based on the user question get context docs
        |
        retriever
        |
        format_docs,
        "question": itemgetter("question")# user question
    }
    |
    qa_prompt # prompt with above user question and context
    |
    llm
)

# Chat Memory
chat_history = StreamlitChatMessageHistory(key="multi_doc_chat")

if len(chat_history.messages) == 0:
    chat_history.add_ai_message("Hi! Ask me anything from your uploaded documents.")
for msg in chat_history.messages:## Render current messages from StreamlitChatMessageHistory
    st.chat_message(msg.type).write(msg.content)

# StreamHandler for LLM Output
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
# Chat Input
if user_question := st.chat_input("Ask your question here..."):
    st.chat_message("human").write(user_question)

    with st.chat_message("ai"):
        container = st.empty()
        handler = StreamHandler()
        handler.container = container  # attach UI

        response = qa_chain.invoke({"question": user_question}, config={"callbacks": [handler]})
        chat_history.add_user_message(user_question)
        chat_history.add_ai_message(handler.text)
