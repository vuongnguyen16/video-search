#luu va tim kiem vecto
"""
searcher.py
Lưu vector vào ChromaDB và tìm kiếm theo cosine similarity.
"""

import chromadb

# ── Khởi tạo DB ──────────────────────────────────────────────────────────────
# PersistentClient: dữ liệu được lưu vào disk, không mất khi tắt máy
_client = chromadb.PersistentClient(path="chroma_db")

# Collection = "bảng" chứa tất cả vector frame
# cosine: phép đo độ tương đồng phù hợp với vector đã normalize của CLIP
_collection = _client.get_or_create_collection(
    name="video_frames",
    metadata={"hnsw:space": "cosine"},
)


# ── Public API ────────────────────────────────────────────────────────────────

def add_to_db(frame_info: dict, embedding: list[float]) -> None:
    """
    Thêm 1 frame vào database.

    Args:
        frame_info: dict với các key: path, video_id, timestamp
        embedding:  vector 512 chiều từ encode_image()
    """
    # ID phải unique — dùng đường dẫn file là đủ
    _collection.add(
        ids=[frame_info["path"]],
        embeddings=[embedding],
        metadatas=[
            {
                "video_id": frame_info["video_id"],
                "timestamp": float(frame_info["timestamp"]),
                "image_path": frame_info["path"],
            }
        ],
    )


def search_db(query_embedding: list[float], top_k: int = 6) -> list[dict]:
    """
    Tìm top_k frame giống query nhất.

    Returns:
        Danh sách dict, mỗi dict gồm:
            - video_id, timestamp, image_path, score (0–1, càng cao càng giống)
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count()),
    )

    frames = []
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance: 0 = giống hệt, 2 = ngược nhau

    for meta, dist in zip(metadatas, distances):
        frames.append(
            {
                "video_id": meta["video_id"],
                "timestamp": meta["timestamp"],
                "image_path": meta["image_path"],
                "score": round(1 - dist / 2, 3),  # chuyển distance → similarity 0–1
            }
        )
    return frames


def db_count() -> int:
    """Số frame đang có trong database."""
    return _collection.count()


def clear_db() -> None:
    """Xóa toàn bộ dữ liệu (dùng khi muốn index lại từ đầu)."""
    _client.delete_collection("video_frames")
    global _collection
    _collection = _client.get_or_create_collection(
        name="video_frames",
        metadata={"hnsw:space": "cosine"},
    )
    print("Đã xóa toàn bộ dữ liệu trong database.")