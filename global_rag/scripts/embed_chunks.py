# File name: embed_chunks.py
# Purpose:
# Generate embeddings for existing chunks and update chunks.embedding.
#
# Input table:
# 1. chunks
#
# Output update:
# 1. chunks.embedding
# 2. chunks.embedding_model
# 3. chunks.search_vector
#
# Notes:
# - Bare-bones POC version.
# - No classes.
# - Uses OpenAI text-embedding-3-small from config.py.
# - Assumes pgvector extension is enabled and chunks.embedding is vector(1536).

import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI

from global_rag.scripts import config


def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    value = str(value).replace("\x00", " ")
    lines = [line.strip() for line in value.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def vector_to_pgvector(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def check_chunks_ready(engine):
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks';
    """

    columns_df = pd.read_sql(text(sql), engine)
    existing_columns = set(columns_df["column_name"].tolist())

    required_columns = [
        "chunk_id",
        "chunk_text",
        "embedding_model",
        "embedding",
        "search_vector",
    ]

    missing_columns = []

    for col in required_columns:
        if col not in existing_columns:
            missing_columns.append(col)

    if missing_columns:
        raise RuntimeError(
            "The chunks table is missing these required columns: "
            + ", ".join(missing_columns)
        )


def embed_chunks():
    config_base = config.config_base()

    db_url = config_base["db_url"]
    openai_api_key = config_base["openai_api_key"]
    embedding_model = config_base["embedding_model"]
    embedding_dimension = int(config_base["embedding_dimension"])

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in Railway environment variables.")

    engine = create_engine(
        url=db_url,
        pool_pre_ping=True
    )

    client = OpenAI(api_key=openai_api_key)

    check_chunks_ready(engine)

    # Keep batch size modest for Railway/API stability.
    batch_size = 50

    chunks_sql = """
        SELECT
            chunk_id,
            chunk_text
        FROM chunks
        WHERE chunk_text IS NOT NULL
          AND LENGTH(TRIM(chunk_text)) > 0
          AND (
                embedding IS NULL
                OR embedding_model IS DISTINCT FROM :embedding_model
              )
        ORDER BY
            document_id,
            chunk_index,
            chunk_id;
    """

    chunks_df = pd.read_sql(
        text(chunks_sql),
        engine,
        params={"embedding_model": embedding_model}
    )

    total_chunks_to_embed = len(chunks_df)
    chunks_embedded = 0

    update_sql = text("""
        UPDATE chunks
        SET
            embedding = CAST(:embedding AS vector),
            embedding_model = :embedding_model,
            search_vector = to_tsvector('english', COALESCE(chunk_text, ''))
        WHERE chunk_id = :chunk_id;
    """)

    for start_index in range(0, total_chunks_to_embed, batch_size):
        batch_df = chunks_df.iloc[start_index:start_index + batch_size].copy()

        input_texts = []

        for _, row in batch_df.iterrows():
            chunk_text = clean_text(row["chunk_text"])

            # Simple safety cap. Your chunking is already around 1,000 tokens,
            # so this should normally not cut anything.
            chunk_text = chunk_text[:30000]

            input_texts.append(chunk_text.replace("\n", " "))

        response = client.embeddings.create(
            model=embedding_model,
            input=input_texts
        )

        update_rows = []

        for item in response.data:
            row = batch_df.iloc[item.index]
            embedding = item.embedding

            if len(embedding) != embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch for chunk_id={row['chunk_id']}. "
                    f"Expected {embedding_dimension}, got {len(embedding)}."
                )

            update_rows.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "embedding": vector_to_pgvector(embedding),
                    "embedding_model": embedding_model,
                }
            )

        with engine.begin() as conn:
            conn.execute(update_sql, update_rows)

        chunks_embedded += len(update_rows)

    remaining_sql = """
        SELECT COUNT(*) AS remaining_chunks
        FROM chunks
        WHERE chunk_text IS NOT NULL
          AND LENGTH(TRIM(chunk_text)) > 0
          AND embedding IS NULL;
    """

    remaining_df = pd.read_sql(text(remaining_sql), engine)
    remaining_chunks = int(remaining_df.loc[0, "remaining_chunks"])

    return {
        "message": "Document embedding completed.",
        "input_table": "chunks",
        "updated_table": "chunks",
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunks_to_embed": int(total_chunks_to_embed),
        "chunks_embedded": int(chunks_embedded),
        "remaining_chunks_without_embedding": remaining_chunks,
        "next_step": "Build retrieval endpoint/query using chunks.embedding for vector search.",
    }


if __name__ == "__main__":
    print(embed_chunks())