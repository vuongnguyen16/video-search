#tai vid tu ytb
"""
downloader.py
Tải video từ YouTube bằng yt-dlp.
"""

import os
import yt_dlp


def download_video(url: str, output_dir: str = "data/videos") -> tuple[str, str, str]:
    """
    Tải 1 video YouTube về thư mục output_dir.

    Returns:
        (video_path, video_id, title)
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        # Ưu tiên mp4 720p để tiết kiệm dung lượng, đủ chất lượng để lấy frame
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        # Tự động merge audio+video nếu cần
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        title = info.get("title", video_id)

        # Tìm đúng tên file đã tải về
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Fallback: tìm file bất kỳ chứa video_id
            for f in os.listdir(output_dir):
                if f.startswith(video_id):
                    video_path = os.path.join(output_dir, f)
                    break

    return video_path, video_id, title