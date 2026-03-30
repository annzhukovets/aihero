"""
Handles data loading, chunking, and vector indexing from GitHub repositories.

Chunks are markdown sections split on headings (see :func:`split_markdown_by_level`).
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import zipfile

import frontmatter
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer


def read_repo_data(repo_owner, repo_name, folder_filter=None):
    """
    Download and parse markdown files from a GitHub repository.

    Same pattern as ``course/app/ingest.py``, plus optional ``folder_filter``
    (e.g. only ``docs/source/en`` for huggingface/transformers).
    """
    url = f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main"
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith(".md") or filename.endswith(".mdx")):
            continue

        if folder_filter and folder_filter.lower() not in filename:
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read().decode("utf-8", errors="ignore")
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split("/", maxsplit=1)
            data["filename"] = filename_repo

            repository_data.append(data)

    zf.close()
    return repository_data


def split_markdown_by_level(text, level=2):
    """
    Split markdown text into sections starting at a given header level.

    Each returned section includes its matching header and all following
    content until the next header of the same level.

    Content before the first matching header is preserved as its own section
    if it is non-empty.
    """
    if level < 1 or level > 6:
        raise ValueError("level must be between 1 and 6")

    pattern = re.compile(
        rf"^\s*(#{{{level}}})\s+(.+?)\s*#*\s*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    sections = []

    first_start = matches[0].start()
    intro = text[:first_start].strip()
    if intro:
        sections.append(intro)

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def chunk_documents(docs, level=2):
    """Split each document on markdown headings; one row per section."""
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        sections = split_markdown_by_level(doc_content, level=level)
        for section_text in sections:
            row = doc_copy.copy()
            row["content"] = section_text
            chunks.append(row)

    return chunks


def _fit_vector_index(docs, embeddings):
    vindex = VectorSearch()
    vindex.fit(embeddings, docs)
    return vindex


def save_artifacts(docs, embeddings, output_dir):
    """
    Save retrieval artifacts to disk:
    - docs.jsonl: one JSON record per line
    - embeddings.npy: numpy matrix with one row per docs record
    """
    os.makedirs(output_dir, exist_ok=True)
    docs_path = os.path.join(output_dir, "docs.jsonl")
    emb_path = os.path.join(output_dir, "embeddings.npy")

    with open(docs_path, "w", encoding="utf-8") as f_out:
        for row in docs:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    np.save(emb_path, np.asarray(embeddings))
    return docs_path, emb_path


def _load_local_artifacts(docs_path, embeddings_path):
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    embeddings = np.load(embeddings_path, allow_pickle=False)
    return docs, embeddings


def _download_artifacts_from_hf(
    repo_id,
    *,
    revision=None,
    subdir="",
    repo_type="dataset",
    token=None,
):
    docs_filename = f"{subdir.strip('/') + '/' if subdir else ''}docs.jsonl"
    emb_filename = f"{subdir.strip('/') + '/' if subdir else ''}embeddings.npy"

    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_path = hf_hub_download(
            repo_id=repo_id,
            filename=docs_filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
        )
        emb_path = hf_hub_download(
            repo_id=repo_id,
            filename=emb_filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
        )
        return _load_local_artifacts(docs_path, emb_path)


def index_data(
    repo_owner,
    repo_name,
    *,
    folder_filter=None,
    doc_filter=None,
    chunk=True,
    level=2,
    embedding_model_name: str = "multi-qa-distilbert-cos-v1",
    artifacts_repo_id: str | None = None,
    artifacts_revision: str | None = None,
    artifacts_subdir: str = "",
    artifacts_repo_type: str = "dataset",
    hf_token: str | None = None,
):
    """
    Download docs, optionally chunk by headings, embed with SentenceTransformers, fit
    :class:`~minsearch.VectorSearch`, and return ``(vector_index, embedding_model)``.

    Queries must be embedded with the same model (see :class:`search_tools.SearchTool`).

    Chunk fields are ``content`` and ``filename``. Heading depth is ``level`` (``##`` when ``level`` is 2).
    """
    embedding_model = SentenceTransformer(embedding_model_name)
    token = hf_token or os.getenv("HF_TOKEN")

    docs = None
    embeddings = None
    if artifacts_repo_id:
        try:
            docs, embeddings = _download_artifacts_from_hf(
                artifacts_repo_id,
                revision=artifacts_revision,
                subdir=artifacts_subdir,
                repo_type=artifacts_repo_type,
                token=token,
            )
            print(f"Loaded prebuilt artifacts from HF Hub: {artifacts_repo_id}")
        except Exception as e:
            print(f"Could not load HF artifacts ({e}); falling back to local build.")

    if docs is None or embeddings is None:
        docs = read_repo_data(repo_owner, repo_name, folder_filter=folder_filter)

        if doc_filter is not None:
            docs = [doc for doc in docs if doc_filter(doc)]

        if chunk:
            docs = chunk_documents(docs, level=level)

        texts = [d["filename"] + " " + d["content"] for d in docs]
        embeddings = embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings)

    vindex = _fit_vector_index(docs, embeddings)
    return vindex, embedding_model
