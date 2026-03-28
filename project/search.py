import numpy as np
from minsearch import Index
from minsearch import VectorSearch


def build_text_index(data, fields):
    """Fit a minsearch Index once; use search_text_index() for each query."""
    data_index = Index(text_fields=fields, keyword_fields=[])
    data_index.fit(data)
    return data_index


def search_text_index(data_index, query, num_results=5):
    return data_index.search(query, num_results=num_results)


def text_search(data, fields, query, num_results=5):
    """One-shot keyword search. Prefer build_text_index + search_text_index when querying repeatedly."""
    data_index = build_text_index(data, fields)
    return search_text_index(data_index, query, num_results=num_results)


def build_vector_index(data, embedding_model, batch_size=32):
    """
    Encode all chunks once and fit a VectorSearch index. Reuse the returned index
    with search_vector_index() for each query (avoids re-encoding documents).
    """
    texts = [d["filename"] + " " + d["section"] for d in data]
    data_embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    data_embeddings = np.asarray(data_embeddings)
    data_vindex = VectorSearch()
    data_vindex.fit(data_embeddings, data)
    return data_vindex


def search_vector_index(vindex, embedding_model, query, num_results=5):
    """Run vector search using a pre-built index; only encodes the query string."""
    q = embedding_model.encode(query)
    return vindex.search(q, num_results=num_results)


def vector_search(data, embedding_model, query, num_results=5):
    """
    One-shot: build index and search. Prefer build_vector_index + search_vector_index
    when you run more than one query against the same chunks.
    """
    vindex = build_vector_index(data, embedding_model)
    return search_vector_index(vindex, embedding_model, query, num_results=num_results)


def hybrid_search(text_results, vector_results, query):
    # Combine and deduplicate results
    seen_ids = set()
    combined_results = []

    for result in text_results + vector_results:
        if result['filename'] not in seen_ids:
            seen_ids.add(result['filename'])
            combined_results.append(result)

    return combined_results
