from langchain.chains import RetrievalQA  #← RetrievalQAをインポートする
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

chat = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data", 
    embedding_function=embeddings
)

retriever = database.as_retriever() #← データベースをRetrieverに変換する

qa = RetrievalQA.from_llm(  #← RetrievalQAを初期化する
    llm=chat,  #← Chat modelsを指定する
    retriever=retriever,  #← Retrieverを指定する
    return_source_documents=True  #← 返答にソースドキュメントを含めるかどうかを指定する
)

result = qa("さつまいもの育て方詳しく")

# print(f'質問({result})')
print(result["result"]) #← 返答を表示する


# print(result["source_documents"]) #← ソースドキュメントを表示する