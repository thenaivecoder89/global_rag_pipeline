import build_document_inventory
import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

config_settings = config.config(project_root="/home/anurag_sarangi/projects/global_rag_pipeline/global_rag/")

origins = config.config(project_root="/home/anurag_sarangi/projects/global_rag_pipeline/global_rag/").allowed_origins

print(origins)