from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, Range, SearchRequest
)
import os

COLLECTION_NAME = "buildings"
VECTOR_SIZE = 768  # jhgan/ko-sroberta-multitask 출력 차원

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )
    return _client


def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[Qdrant] 컬렉션 '{COLLECTION_NAME}' 생성 완료")


def upsert_buildings(points: list[dict]):
    """건물 데이터를 Qdrant에 저장
    points: [{"id": int, "vector": [...], "payload": {building 필드}}]
    """
    client = get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ],
    )


def search(vector: list[float], top_k: int = 20) -> list[dict]:
    """유사도 검색 → payload(건물 정보) + score 반환"""
    client = get_client()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": r.score, "building": r.payload}
        for r in results
    ]
