from pymilvus import MilvusClient
from pymilvus import model
import pandas as pd

client = MilvusClient(
    uri="http://localhost:19530", token="root:Milvus", db_name="default"
)


def load_db():

    client.create_collection(collection_name="tasks", dimension=768)

    embedding_fn = model.DefaultEmbeddingFunction()

    df = pd.read_csv("leetcode-tasks.csv")

    descriptions = [x[:512] for x in df["Problem Description"].to_list()]

    vectors = embedding_fn.encode_documents(descriptions)

    data = [
        {
            "id": i,
            "vector": vectors[i],
            "description": descriptions[i],
            "solution": df.iloc[i, 2],
        }
        for i in range(len(vectors))
    ]

    res = client.insert(collection_name="tasks", data=data)


def semantic_search(query):

    embedding_fn = model.DefaultEmbeddingFunction()

    query_vectors = embedding_fn.encode_queries([query])

    res = client.search(
        collection_name="tasks",
        data=query_vectors,
        limit=1,
        output_fields=["description", "solution"],
    )

    return res


if __name__ == "__main__":

    populate = input("Should the database be populated (Y/N)? ")

    if populate == "Y":
        load_db()

    while True:
        problem = input("Please input the problem: ")

        closest_result = semantic_search(problem)[0][0]["entity"][
            "description"
        ].replace("\\n", "\n")

        print(
            "Here is the closest match we have for your problem: " + str(closest_result)
        )
