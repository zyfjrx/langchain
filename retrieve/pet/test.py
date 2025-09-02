# ------------ 加载文档 ----------------
from langchain.document_loaders import UnstructuredMarkdownLoader
from pprint import pprint

# 2.定义UnstructuredMarkdownLoader对象
md_loader = UnstructuredMarkdownLoader(
    file_path="/Users/zhangyf/Documents/宠物数据/宠物垂直领域大模型数据源-中文/饲养指南-中文/阿比西尼亚猫.md",
    strategy="fast"
)

# 3.加载
docs = md_loader.load()

print(len(docs))
# 4.打印
for doc in docs:
    pprint(doc)
    break


# 1.导入相关依赖
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2.定义RecursiveCharacterTextSplitter分割器对象
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # 每个块中的最大字符数（非单词或Token）
    chunk_overlap=30, # 邻块之间重叠的字符数。重叠的块可以确保如果重要信息横跨两个块，它不会被错过。一般设为chunk_size的10-20%
    length_function=len,
    add_start_index=True,
)

# 3.定义分割的内容


# 4.分割器分割
split_documents = text_splitter.split_documents(docs)

print(len(split_documents))
for doc in split_documents:
    pprint(doc)
    break