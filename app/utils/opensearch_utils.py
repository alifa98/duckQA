import os

import dotenv
from opensearchpy import OpenSearch

dotenv.load_dotenv()

def get_opensearch_client():
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
    OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "user")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "password")
    OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"

    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=False,
        ssl_show_warn=False,
    )
    return client


def create_index(client, index_name, index_body):
    client = get_opensearch_client()

    if not client.indices.exists(index=index_name):
        response = client.indices.create(index=index_name, body=index_body)
        return response
    else:
        return {"acknowledged": True, "message": "Index already exists"}


def index_document(client, index_name, document):
    response = client.index(index=index_name, body=document)
    return response
