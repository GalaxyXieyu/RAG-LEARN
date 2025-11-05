import base64
import os
import uuid
from io import BytesIO
from mimetypes import guess_type

from langchain.retrievers import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.stores import InMemoryStore

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
from datetime import datetime

from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf


from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


app = FastAPI()

os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

# class TimeoutMiddleware(BaseHTTPMiddleware):
#     def __init__(self, app, timeout=120):
#         super().__init__(app)
#         self.timeout = timeout
#
#     async def dispatch(self, request, call_next):
#         try:
#             return await asyncio.wait_for(call_next(request), timeout=self.timeout)
#         except asyncio.TimeoutError:
#             raise HTTPException(status_code=504, detail="Request timeout")
#
#
# app.add_middleware(TimeoutMiddleware, timeout=120)


###################################

class Element(BaseModel):
    type: str
    text: Any


def extract_pdf(pdf_path: str):
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        # Unstructured first finds embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=os.path.dirname(pdf_path)
    )
    return raw_pdf_elements
    # 统计pdf中每种元素类型的个数


def count_pdf_elements(raw_pdf_elements):
    # 创建字典储每种类型的计数
    category_counts = {}

    # 统计每种类型的数量
    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # 打印统计结果
    print("PDF元素类型统计:", category_counts)

    # 全局变量用于存储表格和文本元素


table_elements = []
text_elements = []

## 分类pdf中不同元素element
def categorize_pdf_elements(raw_pdf_elements):
    global table_elements, text_elements
    
    # 创建分类后的元素列表
    categorized_elements = []

    # 根据类型对元素进行分类
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # 使用全局变量存储表格和文本元素
    table_elements.clear()
    table_elements.extend([e for e in categorized_elements if e.type == "table"])
    print(f"表格元素数量: {len(table_elements)}")

    text_elements.clear()
    text_elements.extend([e for e in categorized_elements if e.type == "text"])
    print(f"文本元素数量: {len(text_elements)}")


# 全局变量用于存储摘要
table_summaries = []
text_summaries = []


## 总结表格和文本，形成摘要数组（表格摘要数组 + 文本摘要数组）
def summarize_pdf_elements():
    global table_elements, text_elements, table_summaries, text_summaries
    
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # 修改这里使用Azure OpenAI
    llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",  # or your deployment
            model_version="2024-07-18",
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
    
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    # 处理表格元素
    tables = [i.text for i in table_elements]
    table_summaries.clear()  # 清空现有内容
    table_summaries.extend(summarize_chain.batch(tables, {"max_concurrency": 5}))
    print("表格摘要数量:", len(table_summaries))
    print("表格摘要:", table_summaries)

    # 处理文本元素
    texts = [i.text for i in text_elements]
    text_summaries.clear()  # 清空现有内容
    text_summaries.extend(summarize_chain.batch(texts, {"max_concurrency": 5}))
    print("文本摘要数量:", len(text_summaries))
    print("文本摘要:", text_summaries)


# 全局变量用于存储检索器
retriever = None


def create_retriever():
    global retriever, text_summaries, table_summaries, text_elements, table_elements

    # 修改这里使用Azure OpenAI的embeddings
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",  # Azure上的embeddings署名称
        openai_api_version="2023-05-15",
    )

    vectorstore = Chroma(collection_name="summaries", embedding_function=embeddings)

    # 创建内存存储
    store = InMemoryStore()
    id_key = "doc_id"

    # 初始化检索器
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 获取文本内容，原文本内容数组
    texts = [i.text for i in text_elements]

    # 添加文本及其摘要
    doc_ids = [str(uuid.uuid4()) for _ in texts]

    # 摘要数组，text_summaries转成[Document]数组。Document中有id指向该摘要对应的原文本。
    # 将摘要数组向量化，根据用户问题到摘要向量数据库中找到匹配的摘要，根据id找到对应的原文本，将原文本交给llm回答。
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # 获取表格内容
    tables = [i.text for i in table_elements]

    # 添加表格及其摘要
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    print("检索器创建完成")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

def answer():
    try:
        llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",  # or your deployment
            model_version="2024-07-18",
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )

        # Prompt template
        template = """Answer the question based only on the following context, which can include text and tables:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)


        # RAG pipeline
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke("What is the number of training tokens for LLaMA2?")
        print("LLM回答:", answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask")
async def ask_question():
    try:
        llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",  # or your deployment
            model_version="2024-07-18",
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}.",
                ),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm
        output = chain.invoke(
            {
                "input_language": "English",
                "output_language": "German",
                "input": "I love programming.",
            }
        )

        return {"output": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/process-pdf")
async def process_pdf():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "static", "documents", "LLaMA2.pdf")
    elements = extract_pdf(pdf_path)
    return {"elements": elements}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "static", "documents", "LLaMA2.pdf")
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到PDF文件: {pdf_path}")
        return
    print(f"PDF文件路径: {pdf_path}")
    raw_pdf_elements = extract_pdf(pdf_path)

    print("提取的PDF元素：", raw_pdf_elements)
    count_pdf_elements(raw_pdf_elements)

    ## 分类文本、表格元素 Element
    categorize_pdf_elements(raw_pdf_elements)

    ## LLM生成摘要
    summarize_pdf_elements()

    ## 存储
    create_retriever()

    ## 回答用户问题
    answer()


if __name__ == "__main__":
    main()

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

@app.post("/pdf-image")
async def pdfToImage(file: UploadFile = File(...)):

    # 创建唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{file.filename}"

    # 确保存储目录存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(current_dir, "static", "documents", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # 保存文件路径
    file_path = os.path.join(upload_dir, unique_filename)

    # 保存上传的文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #pdf转image
    images = convert_from_path(file_path, dpi=200, size=(1300,None))  # 降低DPI并限制宽度为800像素,高度自适应
    # 1. 获取第一张图片
    first_page = images[0]

    # 创建临时图片路径
    temp_image_path = os.path.join(upload_dir, f"{timestamp}_temp.png")
    
    # 保存PIL Image到临时文件
    first_page.save(temp_image_path, "PNG")

    base64_image = local_image_to_data_url(temp_image_path)


    # # 2. 将 PIL Image 转换为 base64 字符串
    # buffer = BytesIO()
    # first_page.save(buffer, format="PNG")

    # image_bytes = buffer.getvalue()
    # base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        llm = AzureChatOpenAI(  # 改用 AzureChatOpenAI 而不是 AzureOpenAI
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",
            model_version="2024-07-18",
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个发票图片信息读取专家，擅长读取发票图片中的信息。请按照以下JSON格式输出发票信息:"
            ),
            (
                "human",
                [
                    {"type": "text", "text": "请仔细读取图片中的信息,以严格的json格式输出发票的信息。不要输出任何markdown格式字符。按照输出示例的json格式输出，输出最节省空间的json字符串，不要换行输出。"},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text":
                        """
                        示例输出:
                        {{"发票号码": "24412000000192148949","开票日期": "2024年10月31日", "购买方信息": {{"名称": "考拉的ai树屋","统一社会信用代码": ""}},"销售方信息": {{"名称": "河南乐秦人力资源有限公司","统一社会信用代码": "914111122MADGQEU6D"}},"项目": [{{"名称": "物流辅助服务*派送服务费","规格型号": "","单位": "个", "数量": 1,"单价": 8.49,"金额": 8.49,"税率": "6%","税额": 0.51}}],"合计": {{"金额": 8.49,"税额": 0.51,"总计": 9.00}},"开票人": "沈晨光"}}
                        """
                    }

                    # """ 三重引号，允许字符串跨越多行；在三重引号内，可以自由使用单引号和双引号而不需要转义；
                    # {{ }} 双重括号用于字符串中确实要输出{ 单括号，而不是被理解成{占位符}。这里输出的json字符串中确实存在{，所以需要双重括号。
                ]
            )
        ])

        chain = prompt | llm
        result = chain.invoke({})

        # 解析JSON字符串为Python字典
        import json
        invoice_data = json.loads(result.content)
        
        # 提取发票信息
        invoice_number = invoice_data["发票号码"]
        invoice_date = invoice_data["开票日期"]
        
        # 提取购买方信息
        buyer = invoice_data["购买方信息"]
        buyer_name = buyer["名称"]
        buyer_tax_id = buyer["统一社会信用代码"]
        
        # 提取销售方信息
        seller = invoice_data["销售方信息"] 
        seller_name = seller["名称"]
        seller_tax_id = seller["统一社会信用代码"]
        
        # 提取项目信息
        items = invoice_data["项目"]
        
        # 提取合计信息
        total = invoice_data["合计"]
        total_amount = total["金额"]
        total_tax = total["税额"]
        total_sum = total["总计"]
        
        # 提取开票人
        issuer = invoice_data["开票人"]
        
        # 更新返回结果
        result = {
            "invoice_info": {
                "number": invoice_number,
                "date": invoice_date,
                "issuer": issuer
            },
            "buyer": {
                "name": buyer_name,
                "tax_id": buyer_tax_id
            },
            "seller": {
                "name": seller_name,
                "tax_id": seller_tax_id  
            },
            "items": items,
            "total": {
                "amount": total_amount,
                "tax": total_tax,
                "sum": total_sum
            }
        }
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 检查文件类型
        content_type = file.content_type
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只接受图片文件")
        
        # 创建唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        
        # 确保存储目录存在
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(current_dir, "static", "documents", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存文件路径
        file_path = os.path.join(upload_dir, unique_filename)
        
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 将图片转换为base64
        base64_image = local_image_to_data_url(file_path)

        # 初始化LLM
        llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",
            model_version="2024-07-18", 
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个图片解读专家"
            ),
            (
                "human",
                [
                    {"type": "text", "text": "解读图片"},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
            )
        ])

        chain = prompt | llm
        result = chain.invoke({})

        # 以下写法也是正确的
        # messages = [
        #     {"role": "system", "content": "你是一个图片解读专家"},
        #     {"role": "user", "content": [
        #         {"type": "text", "text": "解读图片"},
        #         {"type": "image_url", "image_url": {
        #             "url": base64_image}
        #          }
        #     ]}
        # ]
        # result = llm.invoke(messages)

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")




@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只接受PDF文件")
        
        # 创建唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        
        # 确保存储目录存在
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(current_dir, "static", "documents", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存文件路径
        file_path = os.path.join(upload_dir, unique_filename)
        
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        raw_pdf_elements = extract_pdf(file_path)
        categorize_pdf_elements(raw_pdf_elements)

        # 处理表格元素
        tables = [i.text for i in table_elements]
        # 处理文本元素
        texts = [i.text for i in text_elements]

        return {
            "tables": tables,
            "texts": texts
        }



        # llm = AzureChatOpenAI(
        #     model="gpt-4o-mini",
        #     azure_deployment="gpt-4o-mini",  # or your deployment
        #     model_version="2024-07-18",
        #     api_version="2024-08-01-preview",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        #     # other params...
        # )
        #
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "你是一个发票信息解读专家，解读 {input} 中的发票信息。去除{input}中换行符 。以严格的json格式输出发票信息，不要其他任何无关字符",
        #         )
        #     ]
        # )
        #
        # chain = prompt | llm
        # output = chain.invoke(
        #     {
        #         "input": text_content,
        #     }
        # )
        #
        #
        # # 返回文件路径供后续处理
        # return {
        #     "text_content": text_content,
        #     "output": output
        # }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


def read_pdf(filepath: str):
    try:
            elements = partition_pdf(
                filename=filepath,
                strategy="hi_res",  # 使用高精度模式
                include_page_breaks=True,  # 包含分页符
            )

            # 只提取文本相关的元素
            text_content = "\n".join([element.text for element in elements if element.text])
            return text_content


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

