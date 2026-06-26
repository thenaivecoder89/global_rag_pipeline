import global_rag.scripts.build_document_inventory as bdi
import global_rag.scripts.config as cnfg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

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
def build_doc_inv(payload: str):
    build_document_inventory_output = bdi.build_document_inventory(payload)
    api_response = JSONResponse(
        {
            "status": "ok",
            "output": build_document_inventory_output
        }
    )
    return api_response