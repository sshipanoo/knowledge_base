from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. 初始化LLM
model_path = "/home/ubuntu/deploy/mistral-q4_0.gguf"  # 替换为您的模型路径
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    temperature=0.7,
    max_tokens=512,
    verbose=True
)

# 2. 初始化VectorStore
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

# 3. 添加文档到VectorStore
documents = [
    "LangChain 是一个用于构建大语言模型应用的框架。",
    "FAISS 是一个高效的向量相似度搜索库。",
    "ChatGLM3-6B 是一个开源的中英双语对话模型。"
]
vectorstore = FAISS.from_texts(documents, embeddings)

# 4. 定义Prompt模板
prompt_template = """基于以下上下文和你的知识，用中文回答：
上下文：{context}
问题：{question}
答案："""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# 5. 初始化RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# 6. 测试问答
query = "告诉我rbd是什么"
result = qa_chain({"query": query})
print("答案：", result["result"])
print("来源：", [doc.page_content for doc in result["source_documents"]])
