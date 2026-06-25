# File_name: config.py
# Purpose: To store all common paths, database settings, model settings,
# and project-level constants used by the RAG pipeline

from pathlib import Path
import os
from dotenv import load_dotenv

# Project paths
project_root = Path(__file__).resolve().parents[1]
corpus_dir =  project_root / "corpus"
client_data_dir = project_root / "client_data"
output_dir = project_root / "output"
draft_report_dir = output_dir / "draft_report"
exceptions_dir = output_dir / "exceptions"
extracted_tables_dir = output_dir / "extracted_tables"
extracted_text_dir = output_dir / "extracted_text"
findings_register_dir = output_dir / "findings_register"

# Database
load_dotenv()
db_url = os.getenv("VECTOR_DB")
if not db_url:
    raise RuntimeError(
        "Database URL variable is not set. Please set to railway DB using public connection string."
    )
document_inv = os.getenv("DOCUMENT_INV")

# LLM
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_provider = "openai"
llm_model = "gpt-5.5" 

# Embeddings
embedding_provider = "openai"
embedding_model = "text-embedding-3-small"
embedding_dimension = 1536

# Chunking
chunk_size_tokens = 1000
chunk_overlap_tokens = 150

# Corpus packs
corpus_packs = {
    "01_RAG_Public_Deal_Documents": corpus_dir / "01_RAG_Public_Deal_Documents",
    "02_Benchmark_and_Market_Data": corpus_dir / "02_Benchmark_and_Market_Data",
    "03_ML_Input_Datasets": corpus_dir / "03_ML_Input_Datasets",
    "04_Seed_Libraries": corpus_dir / "04_Seed_Libraries"
}

# Client data packs
client_data_packs = {
    "TXN_HELIOS_001": client_data_dir / "TXN_HELIOS_001"
}

# File handling
supported_text_extensions = [".txt", ".md"]
supported_document_extensions = [".pdf", ".docx", ".pptx"]
supported_table_extensions = [".csv", ".xlsx", ".xls"]
supported_web_extensions = [".json", ".xml", ".html"]
supported_archive_extensions = [".zip"]
exclude_from_rag = {
    "validation_ground_truth_findings.csv",
    ".DS_Store",
    "Thumbs.db",
    ".identifier"
}
supported_file_types = (
    supported_text_extensions
    + supported_document_extensions
    + supported_table_extensions
    + supported_web_extensions
    + supported_archive_extensions
)
store_raw_files_in_db = False # To avoid storing large files like the ACFE Manual as raw binary in DB

# Corpus folder mapping - used for classifying documents based on folder location
corpus_folder_map = {
    "client_data": "client_evidence",
    "01_RAG_Public_Deal_Documents": "01_RAG_Public_Deal_Documents",
    "02_Benchmark_and_Market_Data": "02_Benchmark_and_Market_Data",
    "03_ML_Input_Datasets": "03_ML_Input_Datasets",
    "04_Seed_Libraries": "04_Seed_Libraries"
}

#  Basic project checks - to confirm required project folders exist
if __name__ == "__main__":
    required_input_folders = [
        client_data_dir,
        corpus_dir,
        client_data_packs["TXN_HELIOS_001"],
        corpus_packs["01_RAG_Public_Deal_Documents"],
        corpus_packs["02_Benchmark_and_Market_Data"],
        corpus_packs["03_ML_Input_Datasets"],
        corpus_packs["04_Seed_Libraries"]
    ]

    for folder in required_input_folders:
        if not folder.exists():
            print(f"WARNING! Folder does not exist: {folder}")
        else:
            print("Folder checks passed!")


# Print configuration summary - 1st if condition added to ensure this codebase
# runs only when executed not when called.
if __name__ == "__main__":
    print("Forensic RAG configuration loaded successfully!")
    print(f"Project root: {project_root}")
    print(f"Client data folder: {client_data_dir}")
    print(f"Corpus data folder: {corpus_dir}")
    print(f"Outputs folder:  {output_dir}")
    print(f"Database URL: {db_url}")
    print(f"Embedding model: {embedding_model}")
    print(f"LLM model: {llm_model}")
    print(f"LLM provider: {llm_provider}")
    print(f"Path for document inventory: {document_inv}")
    if openai_api_key:
        print("Open API key found in environment.")
    else:
        print("WARNING! Open API key not found.")

# Runtime
default_currency = "AED"
default_jurisdiction = "UAE"
project_name = "AI Investment RAG"