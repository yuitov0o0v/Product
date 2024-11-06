from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma
import aiohttp
import asyncio
from langchain.schema import Document
from bs4 import BeautifulSoup

urls = [
    "https://www.takii.co.jp/tsk/manual/daikon.html",
    "https://www.takii.co.jp/tsk/manual/ninjin.html",
    "https://www.takii.co.jp/tsk/manual/kabu.html",
    "https://www.takii.co.jp/tsk/manual/tamanegi.html",
    "https://www.takii.co.jp/tsk/manual/burdock.html",
    "https://www.takii.co.jp/tsk/manual/potato.html",
    "https://www.takii.co.jp/tsk/manual/sweetpotato.html",
    "https://www.takii.co.jp/tsk/manual/kyabetsu.html",
    "https://www.takii.co.jp/tsk/manual/hakusai.html",
    "https://www.takii.co.jp/tsk/manual/broccoli_cauliflower.html",
    "https://www.takii.co.jp/tsk/manual/komatsuna.html",
    "https://www.takii.co.jp/tsk/manual/hourensou.html",
    "https://www.takii.co.jp/tsk/manual/retasu.html",
    "https://www.takii.co.jp/tsk/manual/negi.html",
    "https://www.takii.co.jp/tsk/manual/shungiku.html",
    "https://www.takii.co.jp/tsk/manual/mizunamibuna.html",
    "https://www.takii.co.jp/tsk/manual/celery.html",
    "https://www.takii.co.jp/tsk/manual/tomato.html",
    "https://www.takii.co.jp/tsk/manual/nasu.html",
    "https://www.takii.co.jp/tsk/manual/piman.html",
    "https://www.takii.co.jp/tsk/manual/kyuuri.html",
    "https://www.takii.co.jp/tsk/manual/kabocha.html",
    "https://www.takii.co.jp/tsk/manual/suika.html",
    "https://www.takii.co.jp/tsk/manual/meron.html",
    "https://www.takii.co.jp/tsk/manual/corn.html",
    "https://www.takii.co.jp/tsk/manual/okura.html",
    "https://www.takii.co.jp/tsk/manual/nigauri.html",
    "https://www.takii.co.jp/tsk/manual/edamame.html",
    "https://www.takii.co.jp/tsk/manual/ingen.html",
    "https://www.takii.co.jp/tsk/manual/soramame.html",
    "https://www.takii.co.jp/tsk/manual/endou.html",
    "https://www.takii.co.jp/tsk/manual/strawberry.html"
]

async def fetch_data(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            
            html = await response.text(encoding='shift_jis', errors='ignore')  # エンコーディングをShift-JISに指定
            soup = BeautifulSoup(html, 'html.parser')   # BeautifulSoupでHTMLを解析してテキストを抽出
            text_content = soup.get_text(separator="\n")  # テキストを改行で分けて取得
            return text_content
        else:
            print(f"Error fetching {url}, status code: {response.status}")
            return None

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session: #非同期処理のセッションの再利用し効率化
        tasks = [fetch_data(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def main():
    data = await fetch_all(urls)
    documents = []
    for i, content in enumerate(data):
        if content:
            print(f"Data from URL {i+1} fetched successfully.")
            documents.append(Document(page_content=content, metadata={"source": urls[i]}))
        else:
            print(f"Data from URL {i+1}: failed to fetch")
    return documents

documents = asyncio.run(main())

# 長いコンテンツを事前に分割
MAX_TEXT_LENGTH = 30000
preprocessed_documents = []

for doc in documents:
    content = doc.page_content
    # コンテンツが制限を超えている場合は分割
    if  len(content.encode('shift_jis', errors='ignore')) > MAX_TEXT_LENGTH:
        for i in range(0, len(content), MAX_TEXT_LENGTH):
            preprocessed_documents.append(
                Document(page_content=content[i:i + MAX_TEXT_LENGTH], metadata=doc.metadata)
            )
    else:
        preprocessed_documents.append(doc)

# 事前に分割したドキュメントをSpacyTextSplitterでさらに分割
text_splitter = SpacyTextSplitter(
    chunk_size=300,
    chunk_overlap=150,
    pipeline="ja_core_news_sm"
)
splitted_documents = text_splitter.split_documents(preprocessed_documents)

# OpenAI埋め込みとChromaデータベースを初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
database = Chroma(persist_directory="./.data", embedding_function=embeddings)

# データベースにドキュメントを追加
database.add_documents(splitted_documents)

print("データベースの作成が完了しました。")