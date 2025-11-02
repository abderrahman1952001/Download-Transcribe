
# Streamlit Clien
import os, time, urllib.parse
from urllib.parse import urlsplit
import streamlit as st
from gradio_client import Client, file as gradio_file

st.set_page_config(page_title="Media Toolkit", layout="wide")
st.markdown(
    "<style>.reportview-container .main .block-container{padding-top:1rem; padding-bottom:2rem; max-width: 1200px;} h1,h2,h3{font-weight:700} .stButton>button{border-radius:10px;padding:.5rem 1rem} hr{margin:1.2rem 0}</style>",
    unsafe_allow_html=True
)

def base_root(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    p = urlsplit(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}"
    return url

def ensure_client(url: str) -> Client:
    return Client(base_root(url))

def call_backend(client: Client, api_name: str, *args):
    return client.predict(*args, api_name=f"/{api_name}")

def to_http_file_url(host_root: str, url_or_path: str):
    if not url_or_path or not isinstance(url_or_path, str):
        return None
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    if url_or_path.startswith("file://"):
        path = url_or_path[len("file://"):]
        return f"{host_root}/file={urllib.parse.quote(path)}"
    if url_or_path.startswith("/tmp/gradio/") or url_or_path.startswith("/content/outputs/"):
        return f"{host_root}/file={urllib.parse.quote(url_or_path)}"
    return None

def present_files(files, host_root: str):
    if not files:
        st.info("No files.")
        return
    st.write("**Outputs**")
    for p in files:
        if isinstance(p, dict):
            name = p.get("name") or p.get("orig_name") or p.get("path") or p.get("url") or "file"
            url = p.get("url") or p.get("path") or ""
            http_url = to_http_file_url(host_root, url) if isinstance(url, str) else None
            if not http_url:
                http_url = to_http_file_url(host_root, str(name))
            if http_url:
                shown = os.path.basename(str(name))
                st.write(f"• [{shown}]({http_url})")
            else:
                st.code(str(p))
        else:
            s = str(p)
            http_url = to_http_file_url(host_root, s)
            if http_url:
                name = os.path.basename(s.split("?")[0]) or "file"
                st.write(f"• [{name}]({http_url})")
            else:
                st.code(s)

st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Colab Gradio URL (https://xxxx.gradio.live)", value="")
host_root = base_root(backend_url) if backend_url.strip() else ""
client = None
if backend_url.strip():
    try:
        client = ensure_client(backend_url)
        st.sidebar.success(f"Connected to {host_root}")
        st.sidebar.caption("Use the ROOT URL only.")
    except Exception as e:
        st.sidebar.error(str(e))

st.title("Media Toolkit — Client (v4)")

tab1, tab2, tab3 = st.tabs(["Download", "Transcribe", "OCR"])

with tab1:
    st.subheader("Download")
    st.caption("YouTube URL/ID or playlist → choose voice-only, output type, and quality. Playlists process all items.")
    url = st.text_input("URL or ID", key="dl_url")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"], index=0)
    with c2:
        out_choice = st.selectbox("Output", ["Video", "Audio only"], index=0)
    with c3:
        quality = st.selectbox("Quality (Video only)", ["Best", "1080p", "720p", "480p", "360p"], index=0)
    save_to_drive = st.checkbox("Save to Drive", value=False)
    if st.button("Download", type="primary", disabled=(client is None)):
        if not url.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "download", url, processing, out_choice, quality, bool(save_to_drive))
                st.text_area("Summary", value=log or "", height=180)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")

with tab2:
    st.subheader("Transcribe (single video)")
    st.caption("Arabic lectures (Whisper large-v3). Chunked for long videos. Optional raw TXT.")
    url2 = st.text_input("URL or ID", key="tr_url")
    c1, c2 = st.columns([1,1])
    with c1:
        lang = st.text_input("Language", value="ar")
    with c2:
        save_raw = st.checkbox("Save raw TXT", value=False)
    save_to_drive2 = st.checkbox("Save to Drive", value=False, key="tr_save")
    if st.button("Transcribe", type="primary", disabled=(client is None)):
        if not url2.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "transcribe", url2, lang, bool(save_raw), bool(save_to_drive2))
                st.text_area("Summary", value=log or "", height=180)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")

with tab3:
    st.subheader("OCR")
    st.caption("Text PDFs → pdftotext; scanned PDFs → EasyOCR (GPU auto if available). Batch large PDFs to avoid OOM.")
    pdf = st.file_uploader("PDF file", type=["pdf"])
    c1, c2 = st.columns([1,1])
    with c1:
        dpi = st.slider("DPI", 150, 400, 300, 50)
    with c2:
        batch_pages = st.slider("Batch pages (huge PDFs)", 10, 120, 40, 10)
    save_to_drive3 = st.checkbox("Save to Drive", value=False, key="ocr_save")
    if st.button("Run OCR", type="primary", disabled=(client is None)):
        if not pdf:
            st.warning("Upload a PDF first.")
        else:
            try:
                tmp = f"_upload_{int(time.time())}.pdf"
                with open(tmp, "wb") as f:
                    f.write(pdf.read())
                log, files = call_backend(client, "ocr", gradio_file(tmp), int(dpi), int(batch_pages), bool(save_to_drive3))
                st.text_area("Summary", value=log or "", height=180)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")
