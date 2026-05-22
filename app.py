"""
app.py  —  Giao diện tìm kiếm Streamlit
========================================
Cách chạy:
    streamlit run app.py
"""

import os

import streamlit as st
from PIL import Image

from src.embedder import encode_text, translate_and_encode
from src.searcher import db_count, search_db

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Frame Search",
    page_icon="🎬",
    layout="wide",
)

# ── Tiêu đề ───────────────────────────────────────────────────────────────────
st.title("🎬 Video Frame Search")
st.caption("Tìm kiếm frame video bằng mô tả ngôn ngữ tự nhiên — powered by CLIP")

# ── Sidebar: thông tin DB + cài đặt ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    top_k = st.slider("Số kết quả hiển thị", min_value=3, max_value=12, value=6, step=3)

    lang = st.radio(
        "Ngôn ngữ query",
        ["Tiếng Anh", "Tiếng Việt (tự dịch)"],
        index=0,
        help="Tiếng Anh cho kết quả tốt nhất. Tiếng Việt sẽ tự dịch sang Anh trước.",
    )

    st.divider()
    st.header("📊 Database")
    count = db_count()
    st.metric("Frames đã index", count)
    if count == 0:
        st.warning("Database trống!\nChạy `python pipeline.py` trước.")
    else:
        st.success(f"Sẵn sàng tìm kiếm!")

    st.divider()
    st.caption("Gợi ý query:")
    examples = [
        "person running outdoors",
        "city street traffic",
        "green nature landscape",
        "people talking indoors",
        "close up face emotion",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

# ── Main: ô tìm kiếm ─────────────────────────────────────────────────────────
query = st.text_input(
    "Nhập mô tả cảnh bạn muốn tìm:",
    placeholder="ví dụ: person running in the park",
    key="query_input",
)

search_btn = st.button("🔍 Tìm kiếm", type="primary", disabled=(count == 0))

# ── Xử lý tìm kiếm ───────────────────────────────────────────────────────────
if (search_btn or query) and query.strip():
    if count == 0:
        st.error("Database trống. Vui lòng chạy `python pipeline.py` trước.")
    else:
        with st.spinner("Đang tìm kiếm..."):
            # Encode query theo ngôn ngữ đã chọn
            if lang == "Tiếng Việt (tự dịch)":
                query_vec = translate_and_encode(query)
            else:
                query_vec = encode_text(query)

            results = search_db(query_vec, top_k=top_k)

        if not results:
            st.warning("Không tìm thấy kết quả nào.")
        else:
            st.subheader(f"Top {len(results)} kết quả cho: *\"{query}\"*")
            st.divider()

            # Hiển thị kết quả theo lưới 3 cột
            cols = st.columns(3)
            for idx, r in enumerate(results):
                with cols[idx % 3]:
                    img_path = r["image_path"]
                    if os.path.exists(img_path):
                        st.image(Image.open(img_path), use_container_width=True)
                    else:
                        st.warning(f"Không tìm thấy ảnh:\n`{img_path}`")

                    # Timestamp dạng mm:ss
                    mins = int(r["timestamp"] // 60)
                    secs = int(r["timestamp"] % 60)
                    score_pct = int(r["score"] * 100)

                    st.caption(
                        f"📹 `{r['video_id']}`  "
                        f"⏱ {mins:02d}:{secs:02d}  "
                        f"✨ {score_pct}% match"
                    )

# ── Hướng dẫn khi chưa có query ──────────────────────────────────────────────
if not query:
    st.info(
        "👆 Nhập mô tả vào ô trên để tìm kiếm.\n\n"
        "Ví dụ: `person running`, `blue sky`, `crowd of people`, `night scene`"
    )

    if count > 0:
        st.markdown("---")
        st.subheader("💡 Cách dùng")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Mô tả cảnh vật**")
            st.code("green forest trees\nbeach sunset\nmountain view")
        with col2:
            st.markdown("**Mô tả hành động**")
            st.code("person running\npeople dancing\ncar driving fast")
        with col3:
            st.markdown("**Mô tả cảm xúc / bầu không khí**")
            st.code("happy crowd\ndramatic scene\nquiet empty street")