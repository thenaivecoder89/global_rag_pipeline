# File_name: extract_documents.py
# Purpose: To leverage the document inventory
# and extract the contents as specified.
# This extraction includes:
# - texts from PDFs, PPTs and DOCX files.
# - tables from CSVs

import pandas as pd
from sqlalchemy import create_engine, text

# Import paths and settings from the config file
from global_rag.scripts import config
config_settings = config.config_paths()
config_base = config.config_base()

# Establish DB Connection
engine = create_engine(
    url=config_base["db_url"],
    pool_pre_ping=True
)

# Define query
query = text("select * from build_document_inventory;")

# Pull table data
inventory_df = pd.DataFrame()
with engine.begin() as conn:
    inventory_df = pd.read_sql(
        con=conn,
        sql=query
    )

# Display table data
print(f"Data in build document inventory table:\n{inventory_df.head(5)}")