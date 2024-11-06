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