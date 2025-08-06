# 导入和使用 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import bs4
import os
import dotenv

dotenv.load_dotenv()
loader = WebBaseLoader(
    web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
)
docs = loader.load()
# print(docs)

# 对于嵌入模型，这里通过 API调用
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))

# 使用分割器分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
print(len(documents))
# 向量存储  embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中（默认会存储在内存中）
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()
retriever.search_kwargs = {"k": 3}
docs = retriever.invoke("建设用地使用权是什么？")

# for i,doc in enumerate(docs):
#     print(f"⭐第{i+1}条规定：")
#     print(doc)

# 6.定义提示词模版
prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{info}

用户问：
{question}

请用中文回答用户问题。
"""
# 7.得到提示词模版对象
template = PromptTemplate.from_template(prompt_template)

# 8.得到提示词对象
prompt = template.format(info=docs, question='建设用地使用权是什么？')

## 9. 调用LLM
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url=os.getenv('DASHSCOPE_BASE_URL')
)
response = llm.invoke(prompt)
print(response.content)



from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
# 检索器工具
retriever_tool = create_retriever_tool(
    retriever,
    "CivilCodeRetriever",
    "搜索有关中华人民共和国民法典的信息。关于中华人民共和国民法典的任何问题，您必须使用此工具!",
)

tools = [retriever_tool]



# https://smith.langchain.com/hub
prompt = hub.pull("hwchase17/openai-functions-agent")
# llm = ChatOpenAI(model="qwen-plus", api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'))
# llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 运行代理
res = agent_executor.invoke({"input": "建设用地使用权是什么"})
print(res)
