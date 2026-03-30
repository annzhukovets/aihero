from huggingface_hub import HfApi
from pathlib import Path

import ingest

repo_id = "your-username/transformers-docs-index"  # change this
api = HfApi()
base_dir = Path(__file__).resolve().parent
artifacts_dir = base_dir / "artifacts"

# Build artifacts locally before upload.
repo_owner = "huggingface"
repo_name = "transformers"
docs = ingest.read_repo_data(repo_owner, repo_name, folder_filter="docs/source/en")
docs = ingest.chunk_documents(docs, level=2)
model = ingest.SentenceTransformer("multi-qa-distilbert-cos-v1")
texts = [d["filename"] + " " + d["content"] for d in docs]
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)
docs_path, emb_path = ingest.save_artifacts(docs, embeddings, str(artifacts_dir))
docs_file = Path(docs_path)
emb_file = Path(emb_path)
print(f"Built artifacts: {docs_file} and {emb_file}")

# Create dataset repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# Upload files
api.upload_file(
    path_or_fileobj=str(docs_file),
    path_in_repo="docs.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj=str(emb_file),
    path_in_repo="embeddings.npy",
    repo_id=repo_id,
    repo_type="dataset",
)

print("Uploaded.")