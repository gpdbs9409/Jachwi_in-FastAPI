from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.embedder import embed, embed_batch, building_to_text
from app.vector_store import ensure_collection, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 Qdrant 컬렉션 확인/생성
    ensure_collection()
    yield


app = FastAPI(title="Jachwi-in Embedding Service", lifespan=lifespan)


# ─────────────────────────────────────────────
# Request / Response 모델
# ─────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: list[float]

class SearchRequest(BaseModel):
    query: str          # 사용자 조건 자연어 (예: "CCTV 많고 카페 가까운 곳")
    top_k: int = 20

class SearchResponse(BaseModel):
    results: list[dict]


# ─────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed_text(req: EmbedRequest):
    """단일 텍스트 임베딩 (Spring Boot에서 필요시 호출)"""
    return EmbedResponse(vector=embed(req.text))


@app.post("/search", response_model=SearchResponse)
def search_buildings(req: SearchRequest):
    """사용자 조건 → 유사 건물 검색"""
    vector = embed(req.query)
    results = search(vector, top_k=req.top_k)
    return SearchResponse(results=results)
