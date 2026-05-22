"""
pipeline.py  —  Offline Indexing Pipeline
==========================================
Chạy file này để xử lý tất cả video trong links.xlsx:
    1. Tải video từ YouTube
    2. Trích xuất frame (1 frame / 2 giây)
    3. Encode từng frame bằng CLIP
    4. Lưu vector vào ChromaDB

Cách chạy:
    python pipeline.py
    python pipeline.py --excel links.xlsx --interval 3
"""

import argparse
import os
import time
import re

import pandas as pd

from src.downloader import download_video
from src.embedder import encode_image
from src.extractor import extract_frames
from src.searcher import add_to_db, db_count


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}p{s:02d}s"


def save_status(df: pd.DataFrame, excel_path: str) -> None:
    """Ghi lại trạng thái vào file Excel sau mỗi video."""
    df.to_excel(excel_path, index=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(excel_path: str = "links.xlsx", interval_sec: float = 2.0) -> None:
    """
    Đọc file Excel và index tất cả video có status != 'done'.
    """
    # ── Đọc file Excel ──
    if not os.path.exists(excel_path):
        print(f"[LỖI] Không tìm thấy file: {excel_path}")
        return

    df = pd.read_excel(excel_path)

    # Đảm bảo có cột status và kiểu dữ liệu phù hợp (object) để ghi chuỗi
    if "status" not in df.columns:
        df["status"] = "pending"
    else:
        df["status"] = df["status"].astype(object)
        df["status"].fillna("pending", inplace=True)

    total = len(df)
    pending = df[df["status"] != "done"].shape[0]
    print(f"\n{'='*50}")
    print(f"  Video Semantic Search — Indexing Pipeline")
    print(f"{'='*50}")
    print(f"  File Excel : {excel_path}")
    print(f"  Tổng video : {total}  |  Cần xử lý: {pending}  |  Đã xong: {total - pending}")
    print(f"  Interval   : {interval_sec}s/frame")
    print(f"  DB hiện tại: {db_count()} frames")
    print(f"{'='*50}\n")

    if pending == 0:
        print("Tất cả video đã được index! Chạy app.py để tìm kiếm.")
        return

    start_total = time.time()

    for i, row in df.iterrows():
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", url))
        status = str(row.get("status", "pending")).strip()

        # ── Bỏ qua video đã xử lý ──
        if status == "done":
            print(f"[{i+1}/{total}] Bỏ qua (done): {title}")
            continue

        if not url or url == "nan":
            print(f"[{i+1}/{total}] Bỏ qua (URL trống)")
            df.at[i, "status"] = "skip_empty_url"
            save_status(df, excel_path)
            continue

        print(f"\n[{i+1}/{total}] {title}")
        print(f"  URL: {url}")
        start = time.time()
        video_path = None

        try:
            # Bước 1: Tải video
            print("  [1/3] Đang tải video...")
            video_path, video_id, fetched_title = download_video(url)
            print(f"  Tải xong: {fetched_title} (id={video_id})")

            # Bước 2: Trích frame
            print("  [2/3] Đang trích frame...")
            frames = extract_frames(video_path, video_id, interval_sec=interval_sec)

            # Bước 3: Encode + lưu vào DB
            print(f"  [3/3] Đang encode {len(frames)} frames bằng CLIP...")
            for j, frame in enumerate(frames):
                vec = encode_image(frame["path"])
                add_to_db(frame, vec)
                # In tiến độ mỗi 10 frames
                if (j + 1) % 10 == 0 or (j + 1) == len(frames):
                    print(f"    {j+1}/{len(frames)} frames done", end="\r")
            print()  # xuống dòng sau \r

            elapsed = time.time() - start
            print(f"  Xong! {len(frames)} frames — {format_time(elapsed)}")

            # Đánh dấu done
            df.at[i, "status"] = "done"
            save_status(df, excel_path)

        except Exception as e:
            # Làm sạch thông báo lỗi: loại bỏ mã ANSI và ký tự điều khiển
            raw = str(e)
            raw = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', raw)  # remove ANSI
            raw = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw)     # remove control chars
            error_msg = raw[:80]   # giới hạn độ dài để vừa Excel
            print(f"  [LỖI] {error_msg}")
            df.at[i, "status"] = f"error: {error_msg}"
            save_status(df, excel_path)

        finally:
            # Xóa file video để tiết kiệm dung lượng
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"  Đã xóa file video tạm.")

        # Nghỉ giữa các video để tránh bị YouTube giới hạn
        time.sleep(1)

    # ── Tổng kết ──
    total_time = time.time() - start_total
    done_count = df[df["status"] == "done"].shape[0]
    error_count = df[df["status"].str.startswith("error", na=False)].shape[0]

    print(f"\n{'='*50}")
    print(f"  Hoàn tất!")
    print(f"  Thành công : {done_count}/{total}")
    print(f"  Lỗi       : {error_count}/{total}")
    print(f"  Tổng thời gian: {format_time(total_time)}")
    print(f"  DB hiện tại   : {db_count()} frames")
    print(f"{'='*50}")
    print(f"\nChạy 'streamlit run app.py' để mở giao diện tìm kiếm!\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index video YouTube vào vector DB")
    parser.add_argument(
        "--excel", default="links.xlsx", help="Đường dẫn file Excel (mặc định: links.xlsx)"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Khoảng cách giữa các frame tính bằng giây (mặc định: 2.0)"
    )
    args = parser.parse_args()
    run(excel_path=args.excel, interval_sec=args.interval)
