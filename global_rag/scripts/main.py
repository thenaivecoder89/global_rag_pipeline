import build_document_inventory
import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

app = FastAPI()

config_base = config.config_base()
origins = config_base["allowed_origins"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get(path="/build_document_inventory", status_code=200)
def build_doc_inv(payload: str):
    build_document_inventory_output = build_document_inventory(payload)
    api_response = JSONResponse(
        {
            "status": "ok",
            "output": build_document_inventory_output
        }
    )
    return api_response