#trich frame tu video
"""
extractor.py
Trích xuất frame từ video bằng OpenCV.
"""

import os
import cv2


def extract_frames(
    video_path: str,
    video_id: str,
    interval_sec: float = 2.0,
    output_dir: str = "data/frames",
) -> list[dict]:
    """
    Trích 1 frame mỗi interval_sec giây từ video.

    Returns:
        Danh sách dict, mỗi dict gồm:
            - path: đường dẫn file ảnh
            - video_id: ID video YouTube
            - timestamp: thời điểm (giây)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback nếu video không báo fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * interval_sec))

    saved = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            timestamp_sec = frame_idx / fps
            # Tên file: video_id + timestamp 5 chữ số để dễ sort
            filename = os.path.join(
                output_dir, f"{video_id}_t{int(timestamp_sec):05d}.jpg"
            )
            # IMWRITE_JPEG_QUALITY 85: cân bằng chất lượng vs dung lượng
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved.append(
                {
                    "path": filename,
                    "video_id": video_id,
                    "timestamp": timestamp_sec,
                }
            )

        frame_idx += 1

    cap.release()

    duration_min = (total_frames / fps) / 60
    print(
        f"  Trích {len(saved)} frames "
        f"(video {duration_min:.1f} phút, {fps:.0f}fps, "
        f"1 frame/{interval_sec}s)"
    )
    return saved