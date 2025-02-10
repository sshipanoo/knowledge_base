# 后端修改（main.py）
import os
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 配置路径
UPLOAD_FOLDER = './uploaded_files'
FAISS_INDEX_DIR = './faiss_index'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

# 初始化向量数据库
if os.path.exists(FAISS_INDEX_DIR):
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings,allow_dangerous_deserialization=True)
else:
    loader = DirectoryLoader('./docs', glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

# 自定义回调处理流式输出
class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
    
    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put({"data": token})
    
    async def on_chain_end(self, outputs, **kwargs):
        await self.queue.put({"sources": [doc.page_content for doc in outputs["source_documents"]]})
        await self.queue.put(None)  # 结束信号

# 初始化LLM
model_path = "/home/ubuntu/deploy/mistral-q4_0.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    temperature=0.7,
    max_tokens=512,
    streaming=True,  # 启用流式支持
    verbose=True
)

# 定义Prompt模板
prompt_template = """完全基于以下上下文和你的知识，用中文回答：
上下文：{context}
问题：{question}
答案："""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# 文件上传接口（修复版）
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, detail="仅支持txt文件")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        # 保存文件
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # 处理上传文件
        loader = TextLoader(file_path)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？"]
        )
        splits = text_splitter.split_documents(document)
        
        # 更新索引
        vectorstore.add_documents(splits)
        vectorstore.save_local(FAISS_INDEX_DIR)
        
        return {"filename": file.filename, "message": "文件处理成功"}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=f"处理失败: {str(e)}")

# 流式问答接口
@app.post("/ask")
async def ask(question: str = Form(...)):
    callback = StreamCallbackHandler()
    
    async def event_generator():
        # 在后台运行问答链
        task = asyncio.create_task(
            qa_chain.acall(
                {"query": question},
                callbacks=[callback]
            )
        )
        
        while True:
            item = await callback.queue.get()
            if item is None:
                break
            if "data" in item:
                yield f"data: {item['data']}\n\n"
            elif "sources" in item:
                yield f"event: sources\ndata: {item['sources']}\n\n"
        await task
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 首页
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
