import time
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    db,
)
import pyperclip
from polyglot.text import Text
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

def segmentSentence(text):
    """
    对文本进行分句
    :param text: 待分句的文本
    :return: 分句后的文本列表
    """
    sentences = Text(text).sentences
    print(sentences)
    return sentences

def getEmbeddingsToDatas(sentences, model):
    """
    获取文本的向量表示
    :param sentences: 待获取向量表示的文本
    :param model: 模型
    :return: 向量表示的文本列表
    """ 
    datas = []
    if type(sentences) == list:
        for sentence in sentences:
            embeddings = model.encode(sentence)
            datas.append([sentence, embeddings.tolist()])
    else:
            embeddings = model.encode(sentences)
            datas.append([sentences, embeddings.tolist()])
    return datas



def addDataToMilvus(db_name, table_name, datas):
    """
    将数据添加到Milvus数据库中
    :param db_name: 数据库名称
    :param table_name: 表格名称
    :param datas: 待添加的数据
    :return: None
    """
    client = MilvusClient(
        db_name=db_name,
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    formatted_data = [
    {
        "id": idx,  # 添加id
        "sequence": str(data[0]),
        "embeddings": data[1]
    }
    for idx, data in enumerate(datas)
    ]
    print(formatted_data)
    res = client.insert(
        collection_name=table_name,
        data=formatted_data
    )
    print(res)

def searchMilvusSentences(queryEembeddings, db_name, table_name, downLine, upLine, limitNum):
    """
    在Milvus数据库中搜索相似的句子
    :param queryEembeddings: 待搜索的向量表示
    :param db_name: 数据库名称
    :param table_name: 表格名称
    :param downLine: 下限
    :param upLine: 上限
    :param limitNum: 限制返回的条数
    :return: 搜索结果
    """
    client = MilvusClient(
        db_name=db_name,
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    result = client.search(
    collection_name=table_name,
    data=[queryEembeddings],
    limit=limitNum,
    search_params={
        "params": {
            "radius": downLine,
            "range_filter": upLine

        }
    },
    output_fields=["id","sequence"],
    )
    print(result[0])
    print("我的内容是{}".format(len(result[0])))

    return result[0]
    # for hits in result:
        # print("TopK results:")
        # for hit in hits:
        #     # print("[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]")
        #     # print(hit["entity"][0])
        #     if hit == "['[]']":
        #         print("No match found")
        #     else:
        #         print(hit["entity"][0])
        #         result = hit["entity"][0]
        #         print(1)

def dealOnesentence(text):
    """
    处理单个句子
    :param text: 待处理的句子
    :return: None
    """
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='D:/huggingface/hub', trust_remote_code=True)
    data = model.encode(str(text))
    searchMilvusSentences(queryEembeddings=data, db_name="textTogame", table_name="newAllPrompt", downLine=0.4, upLine=1.1, limitNum=5)
    

def dealAllsentences(text):
    sentences = segmentSentence(text)
    dealSentences = ""
    preSentence = sentences[0]
    start_time = time.time()
    for sentence in sentences:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='D:/huggingface/hub', trust_remote_code=True)
        data = model.encode(str(sentence))
        print(sentence)
        embeddingTime = time.time()
        print(f"embeddingTime: {embeddingTime - start_time:.2f} 秒")
        resultsZeroSenven = searchMilvusSentences(queryEembeddings=data, db_name="textTogame", table_name="newAllPrompt", downLine=0.7, upLine=1.1, limitNum=1)
        resultsZeroSix = searchMilvusSentences(queryEembeddings=data, db_name="textTogame", table_name="newAllPrompt", downLine=0.6, upLine=1.1, limitNum=2)
        resultsZerofive = searchMilvusSentences(queryEembeddings=data, db_name="textTogame", table_name="newAllPrompt", downLine=0.5, upLine=1.1, limitNum=3)
        resultsZerofour = searchMilvusSentences(queryEembeddings=data, db_name="textTogame", table_name="newAllPrompt", downLine=0.4, upLine=1.1, limitNum=7)
        if len(resultsZeroSenven) == 1 or len(resultsZeroSix) == 2 or len(resultsZerofive) == 3 or (
            len(resultsZeroSix)==1 and len(resultsZerofive) == 2 and len(resultsZerofour) == 5) or (
            len(resultsZeroSix)==0 and len(resultsZerofive)==2 and len(resultsZerofour) == 7):
            flag=True
        else:
            flag=False
        if flag == False:
            dealSentences += str(sentence)
            preSentence = sentence
        else:
            if preSentence == "涉嫌违规词语":
                dealSentences += ""
                preSentence = "涉嫌违规词语"
            else:
                dealSentences += "涉嫌违规词语"
                preSentence = "涉嫌违规词语"
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间：{elapsed_time:.2f} 秒")
    return dealSentences

if __name__ == '__main__':
    # results = searchMilvusSentences(db_name="textTogame", table_name="newPrompt")
    # sequencesList = []
    # for result in results:
    #     sequencesList.append(result["sequence"])
    sequencesList = [
    "You are a professional front-end development engineer and you are very good at using the Phaser3 framework.",
    "You need to solve the user's question and reply in the form of code, without any placeholders and non-code content.",
    "Your answer is always complete code, no placeholders are needed, and the full implementation needs to be included in each method.",
    "Please note that the code should be completely feasible.",
    "There are no placeholders, no to-do items, and make sure that all the code can run correctly in the production environment.",
    "You need to carefully analyze the current code structure and the processing logic of each method, and then modify the user's problem, only need to modify the part of the code related to the user's problem, to ensure that the rest of the original logic is valid.",
    "If a new image or audio is needed to resolve a user issue and the user does not indicate it, use any of the following links to take up space, but do not duplicate existing ones.",
    "Preload all images and audio used in the game within the 'preload' function.",
    "Do not place the 'load' method inside loops or other special structures.",
    "If there is logic involved in restarting the game, and the user has no special needs, you need to make sure that after restarting the game, the background music will overlap and the user will not be able to control the error situation.",
    "If you need to explain the modified content, the explanation will be put before the complete code, and will be returned in strict accordance with the order of 'modify the explanation + modified complete code', and the modified complete code needs to be wrapped with html code blocks."
    ]
    print(len(sequencesList))
    combinations = []
    for i in range(len(sequencesList)):
        for j in range(i + 1, len(sequencesList) + 1):
            combinations.append(''.join(sequencesList[i:j]))
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='D:/huggingface/hub', trust_remote_code=True)
    data = getEmbeddingsToDatas(combinations, model)
    addDataToMilvus(db_name="textTogame", table_name="veryNewPrompt", datas=data)
    # results = searchMilvusSentences(db_name="textTogame", table_name="allPrompt")
    # print(results)
    # start_time = time.time()
    # text = """
    
    # """
    # dealSentence = dealAllsentences(text)
    # pyperclip.copy(dealSentence)

    # text = "You are a professional front-end development engineer and you are very good at using the Phaser3 framework."
    # text1 = "Summary of rules:  I must provide complete, runnable Phaser3 code with no placeholders, load all assets in the preload function, and explain any modifications."
    # text2 = "I must always respond with code only, within markdown code blocks, and end each response with "### code end"."
    # text3 = "Please note that the code should be completely feasible."
    # text4 = "Placeholder image and audio - replace with your actual assets"
    # print(text1)
    # dealOnesentence(text1)
    # print(text2)
    # dealOnesentence(text2)
    # print(text3)
    # dealOnesentence(text3)

    # print(text4)
    # dealOnesentence(text4)
