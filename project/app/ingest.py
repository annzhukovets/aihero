"""
Handles data loading, chunking, and vector indexing from GitHub repositories.

Chunks are markdown sections split on headings (see :func:`split_markdown_by_level`).
"""
from __future__ import annotations

import io
import re
import zipfile

import frontmatter
import numpy as np
import requests
from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer


def read_repo_data(repo_owner, repo_name, folder_filter=None):
    """
    Download and parse markdown files from a GitHub repository.

    Same pattern as ``course/app/ingest.py``, plus optional ``folder_filter``
    (e.g. only ``docs/source/en`` for huggingface/transformers).
    """
    url = f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main"
    resp = requests.get(url, timeout=60)

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


def index_data(
    repo_owner,
    repo_name,
    *,
    folder_filter=None,
    doc_filter=None,
    chunk=True,
    level=2,
    embedding_model_name: str = "multi-qa-distilbert-cos-v1",
    max_chunks: int | None = None,
):
    """
    Download docs, optionally chunk by headings, embed with SentenceTransformers, fit
    :class:`~minsearch.VectorSearch`, and return ``(vector_index, embedding_model)``.

    Queries must be embedded with the same model (see :class:`search_tools.SearchTool`).

    Chunk fields are ``content`` and ``filename``. Heading depth is ``level`` (``##`` when ``level`` is 2).
    """
    docs = read_repo_data(repo_owner, repo_name, folder_filter=folder_filter)

    if doc_filter is not None:
        docs = [doc for doc in docs if doc_filter(doc)]

    if chunk:
        docs = chunk_documents(docs, level=level)

    # Cap the number of chunks so Streamlit startup doesn't hang/timeout.
    if max_chunks is not None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0 when provided")
        docs = docs[:max_chunks]

    embedding_model = SentenceTransformer(embedding_model_name)
    texts = [d["filename"] + " " + d["content"] for d in docs]
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings)

    vindex = VectorSearch()
    vindex.fit(embeddings, docs)
    return vindex, embedding_model
