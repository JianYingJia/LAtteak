from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    db,
)

def createCollection(db_name, collection_name):
    """
    创建collection
    :param db_name: 数据库名称
    :param collection_name: collection名称
    :return: collection状态
    """
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus", db_name=db_name)
    # print(client)
    schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="sentence", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
    client.create_collection(collection_name, schema)
    result = client.get_load_state(
    collection_name=collection_name
    )

    return result

def createMilvusTable(db_name, table_name):
    """
    创建milvus表
    :param db_name: 数据库名称
    :param table_name: 表名称
    :return: 表状态
    """
    client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name=db_name
    )

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="sequence", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=768)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )
    index_params.add_index(
        field_name="embeddings", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    client.create_collection(
    collection_name=table_name,
    schema=schema,
    index_params=index_params,
    )
    result = client.get_load_state(
        collection_name=table_name
    )
    return result
