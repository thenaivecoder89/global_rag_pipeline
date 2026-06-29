import global_rag.scripts.build_document_inventory as bdi
import global_rag.scripts.config as cnfg
import global_rag.scripts.extract_documents as ed
import global_rag.scripts.chunk_documents as cd
import global_rag.scripts.embed_chunks as emb
import global_rag.scripts.retrieve_chunks as ret
import global_rag.scripts.report_generation as rg

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from pathlib import Path

app = FastAPI()

config_base = cnfg.config_base()
origins = config_base["allowed_origins"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok", "message": "FastAPI service is running"}

@app.get(path="/build_document_inventory", status_code=200)
def build_doc_inv():
    build_document_inventory_output = bdi.build_document_inventory()
    api_response = JSONResponse(
        {
            "status": "ok",
            "output": build_document_inventory_output
        }
    )
    return api_response

@app.get("/debug_paths")
def debug_paths():
    base = Path("/app")

    return {
        "app_exists": base.exists(),
        "app_children": [str(p) for p in base.iterdir()] if base.exists() else [],
        "cwd": str(Path.cwd()),
    }

@app.get(path="/extract_documents", status_code=200)
def extract_docs():
    extract_documents_output = ed.extract_documents()

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": extract_documents_output
        }
    )

    return api_response

@app.get(path="/chunk_documents", status_code=200)
def chunk_docs():
    chunk_documents_output = cd.chunk_documents()

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": chunk_documents_output
        }
    )

    return api_response

@app.get(path="/embed_chunks", status_code=200)
def embed_chunks():
    embed_documents_output = emb.embed_chunks()

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": embed_documents_output
        }
    )

    return api_response

@app.get(path="/retrieve_chunks", status_code=200)
def retrieve_chunks_api(
    q: str,
    top_k: int = 10,
    mode: str = "hybrid",
    corpus_zone: Optional[str] = None,
    corpus_pack: Optional[str] = None,
    document_id: Optional[str] = None,
    max_chunk_chars: int = 3000
):
    retrieval_output = ret.retrieve_chunks(
        query_text=q,
        top_k=top_k,
        mode=mode,
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": retrieval_output
        }
    )

    return api_response

@app.get(path="/generate_ic_review_report", status_code=200)
def generate_ic_review_report_api(
    transaction_id: str = "TXN_HELIOS_001",
    use_llm_summary: bool = True,
    write_audit: bool = True
):
    report_generation_output = rg.generate_investment_ic_review_report(
        transaction_id=transaction_id,
        use_llm_summary=use_llm_summary,
        write_audit=write_audit
    )

    api_response = JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": report_generation_output
            }
        )
    )

    return api_response