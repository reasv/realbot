import asyncio
import base64
import mimetypes
import os
import re
import uuid
from typing import Optional

import requests

IMAGES_ROOT = os.path.join("history", "images")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def ensure_channel_image_dir(channel_id: int | str) -> str:
    """
    Ensures the on-disk directory for a channel's stored images exists.
    """
    path = os.path.join(IMAGES_ROOT, str(channel_id))
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(filename: str, fallback_prefix: str = "image") -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", filename).strip("._")
    return cleaned or f"{fallback_prefix}_{uuid.uuid4().hex}"


async def download_image_to_history(
    channel_id: int | str, url: str, filename: Optional[str] = None
) -> str:
    """
    Downloads an image from `url` into the channel's history image directory.
    Returns the relative path to the stored file.
    """
    if not filename:
        filename = url.split("?")[0].split("/")[-1] or "image"
    filename = sanitize_filename(filename)
    directory = ensure_channel_image_dir(channel_id)
    target_path = os.path.join(directory, filename)

    # Ensure uniqueness
    if os.path.exists(target_path):
        stem, ext = os.path.splitext(filename)
        target_path = os.path.join(directory, f"{stem}_{uuid.uuid4().hex}{ext}")

    def _download():
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(resp.content)

    await asyncio.to_thread(_download)
    return target_path


def load_image_as_data_url(path: str) -> str:
    """
    Loads the given image from disk and returns a data URL suitable for OpenAI input.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_supported_image_mime(content_type: Optional[str], filename: Optional[str] = None) -> bool:
    """
    Returns True if the provided mime type or filename looks like an image we should process.
    """
    if content_type and content_type.startswith("image/"):
        return True
    if filename:
        extension = (filename.rsplit(".", 1)[-1] or "").lower()
        return extension in {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
    return False


def build_remote_image_record(url: str, source: str, description: Optional[str] = None) -> dict:
    """
    Helper that returns the standard metadata structure for remote-only images.
    """
    record: dict = {"source": source, "url": url}
    if description:
        record["description"] = description
    return record
