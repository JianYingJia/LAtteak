from sentence_transformers import SentenceTransformer

def textToembedding(text):
    model = SentenceTransformer(r'D:\huggingface\hub\models--sentence-transformers--all-mpnet-base-v2\snapshots\9a3225965996d404b775526de6dbfe85d3368642', cache_folder='D:/huggingface/hub')
    embeddings = model.encode(str(text))
    return embeddings