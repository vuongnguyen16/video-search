# Video Frame Search

Tìm kiếm frame video bằng câu mô tả ngôn ngữ tự nhiên — powered by CLIP + ChromaDB.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách dùng

**Bước 1:** Thêm link YouTube vào `links.xlsx` (cột `url`).

**Bước 2:** Chạy pipeline để index video:
```bash
python pipeline.py
```

**Bước 3:** Mở giao diện tìm kiếm:
```bash
streamlit run app.py
```

## Cấu trúc thư mục

```
video-search/
├── src/
│   ├── downloader.py   # tải video từ YouTube
│   ├── extractor.py    # trích xuất frame
│   ├── embedder.py     # CLIP encode ảnh + text
│   └── searcher.py     # ChromaDB lưu + tìm kiếm
├── data/
│   ├── videos/         # video tạm (tự xóa sau khi xử lý)
│   └── frames/         # frame ảnh đã trích xuất
├── chroma_db/          # vector database
├── app.py              # giao diện Streamlit
├── pipeline.py         # offline indexing pipeline
├── links.xlsx          # danh sách link YouTube
└── requirements.txt
```
