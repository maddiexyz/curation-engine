import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch


def load_clip_model(
    model_name: str = "clip-ViT-B-32",
    device: Optional[str] = None,
) -> SentenceTransformer:
    """Loads and returns a CLIP model instance.

    Args:
        model_name: Hugging Face or sentence-transformers model identifier.
        device: 'cuda', 'cpu', or None (auto-detects CUDA availability).
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(model_name, device=target_device)


def embed_text(
    model: SentenceTransformer,
    text: Union[str, List[str]],
    normalize: bool = True,
) -> Union[List[float], List[List[float]]]:
    """Computes embeddings for a single text query or a list of queries.

    Args:
        model: Loaded SentenceTransformer CLIP instance.
        text: String or list of strings to encode.
        normalize: If True, produces unit-length vectors (recommended for cosine similarity).

    Returns:
        A list of floats for a single string, or a list of lists of floats for batches.
    """
    is_single = isinstance(text, str)
    inputs = [text] if is_single else text

    embeddings = model.encode(
        inputs,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if is_single:
        return embeddings[0].tolist()
    return [vec.tolist() for vec in embeddings]


def embed_image(
    model: SentenceTransformer,
    image_source: Union[Image.Image, str, Path, bytes, BinaryIO],
    normalize: bool = True,
) -> List[float]:
    """Computes a 512-dimension visual embedding from an in-memory PIL image, path, or bytes.

    Args:
        model: Loaded SentenceTransformer CLIP instance.
        image_source: PIL Image object, file path, raw bytes, or file-like buffer.
        normalize: If True, produces unit-length vectors (recommended for cosine similarity).

    Returns:
        List of floats representing the image embedding vector.
    """
    if isinstance(image_source, Image.Image):
        pil_image = image_source.convert("RGB")
    elif isinstance(image_source, (str, Path)):
        pil_image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        pil_image = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif hasattr(image_source, "read"):
        pil_image = Image.open(image_source).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")

    embedding = model.encode(
        pil_image,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embedding.tolist()