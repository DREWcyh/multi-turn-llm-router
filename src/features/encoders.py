import numpy as np
import os
def load_encoder(model_name):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("need sentence-transformers, please install requirements.txt") from e
    return SentenceTransformer(model_name)
def encode_texts(model, texts, batch_size=64):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)
