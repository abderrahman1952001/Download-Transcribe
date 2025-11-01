# Streamlit client (FINAL) — 
import os, time
from urllib.parse import urlsplit
import streamlit as st
from gradio_client import Client, file as gradio_file

st.set_page_config(page_title="Media Toolkit Client", layout="wide")

# --------- Tiny style polish (safe) ----------
st.markdown("""
<style>
/* Simple, safe tweaks */
.reportview-container .main .block-container{padding-top:1rem; padding-bottom:2rem; max-width: 1200px;}
h1, h2, h3 { font-weight: 700; }
.stButton>button { border-radius: 10px; padding: 0.5rem 1rem; }
hr { margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# --------- Helpers ----------
def base_root(url: str) -> str:
    """
    Extract https://host from a Gradio URL.
    Accepts full URLs or already-rooted hosts.
    """
    url = (url or "").strip().rstrip("/")
    p = urlsplit(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}"
    return url

def ensure_client(url: str) -> Client:
    return Client(base_root(url))

def call_backend(client: Client, api_name: str, *args):
    # We hit the named Gradio endpoints: /download, /transcribe, /ocr
    return client.predict(*args, api_name=f"/{api_name}")

def to_http_file_url(host_root: str, url_or_path: str):
    """
    Convert Gradio's local file paths/URLs into publicly reachable URLs.
    Examples:
      file:///tmp/gradio/abcd/file.mp4 -> https://<host>/file=/tmp/gradio/abcd/file.mp4
      /tmp/gradio/abcd/file.mp4        -> https://<host>/file=/tmp/gradio/abcd/file.mp4
      http(s) URLs returned by Gradio are passed through.
    """
    if not url_or_path or not isinstance(url_or_path, str):
        return None
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    if url_or_path.startswith("file://"):
        return f"{host_root}/file={url_or_path[len('file://'):]}"
    if url_or_path.startswith("/tmp/gradio/"):
        return f"{host_root}/file={url_or_path}"
    return None

def present_files(files, host_root: str):
    """
    Render outputs as clickable links. We rewrite local paths/URLs to the
    server's /file= route so they work outside the backend.
    """
    if not files:
        st.info("No files.")
        return
    st.write("**Outputs**")
    for p in files:
        # Gradio often returns dictionaries with fields like 'name', 'url', 'path'
        if isinstance(p, dict):
            name = p.get("name") or p.get("orig_name") or p.get("path") or p.get("url") or "file"
            url = p.get("url") or p.get("path") or ""
            http_url = to_http_file_url(host_root, url) if isinstance(url, str) else None
            if not http_url:
                # Try converting the name if it's a path-like
                http_url = to_http_file_url(host_root, str(name))
            if http_url:
                st.write(f"• [{os.path.basename(str(name))}]({http_url})")
            else:
                # last resort: show raw object
                st.code(str(p))
        else:
            s = str(p)
            http_url = to_http_file_url(host_root, s)
            if http_url:
                st.write(f"• [{os.path.basename(s.split('?')[0]) or 'file'}]({http_url})")
            else:
                st.code(s)

# --------- Sidebar: backend connection ----------
st.sidebar.header("Backend")
backend_url = st.sidebar.text_input(
    "Colab Gradio URL (root only, e.g. https://xxxx.gradio.live)",
    value=""
)
host_root = base_root(backend_url) if backend_url.strip() else ""
client = None
if backend_url.strip():
    try:
        client = ensure_client(backend_url)
        st.sidebar.success(f"Connected to {host_root}")
        st.sidebar.caption("Tip: The URL must be the **root** (no /download etc.).")
    except Exception as e:
        st.sidebar.error(str(e))

# --------- Main ----------
st.title("Media Toolkit — Client")

tab1, tab2, tab3 = st.tabs(["Download", "Transcribe", "OCR"])

with tab1:
    st.subheader("Download")
    st.caption("Paste a YouTube URL/ID or playlist. Choose optional music removal (voice-only), output type, and quality.")
    url = st.text_input("URL or ID", key="dl_url")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"], index=0)
    with c2:
        out_choice = st.selectbox("Output", ["Video", "Audio only"], index=0)
    with c3:
        quality = st.selectbox("Quality (Video only)", ["Best", "1080p", "720p", "480p", "360p"], index=0)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], index=0)
    run_dl = st.button("Download", type="primary", disabled=(client is None))
    if run_dl:
        if not url.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "download", url, processing, out_choice, quality, dest_choice)
                st.text_area("Log", value=log or "", height=220)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")

with tab2:
    st.subheader("Transcribe (single video)")
    st.caption("Arabic lectures supported (Whisper large-v3). Optional English translation and raw TXT export.")
    url2 = st.text_input("URL or ID", key="tr_url")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        lang = st.text_input("Language", value="ar")
    with c2:
        trans = st.checkbox("Translate to English", value=False)
    with c3:
        save_raw = st.checkbox("Save raw TXT", value=False)
    dest2 = st.selectbox("Destination", ["Here", "Drive"], index=0, key="tr_dest")
    run_tr = st.button("Transcribe", type="primary", disabled=(client is None))
    if run_tr:
        if not url2.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "transcribe", url2, lang, bool(trans), bool(save_raw), dest2)
                st.text_area("Log", value=log or "", height=220)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")

with tab3:
    st.subheader("OCR")
    st.caption("Text PDFs use pdftotext fast-path when available; scanned PDFs are OCR’d with Arabic-friendly settings.")
    pdf = st.file_uploader("PDF file", type=["pdf"])
    c1, c2, _ = st.columns([1,1,1])
    with c1:
        dpi = st.slider("DPI", 150, 400, 300, 50)
    with c2:
        batch_pages = st.slider("Batch pages (for huge PDFs)", 10, 120, 40, 10)
    dest3 = st.selectbox("Destination", ["Here", "Drive"], index=0, key="ocr_dest")
    run_ocr = st.button("Run OCR", type="primary", disabled=(client is None))
    if run_ocr:
        if not pdf:
            st.warning("Upload a PDF first.")
        else:
            try:
                tmp = f"_upload_{int(time.time())}.pdf"
                with open(tmp, "wb") as f:
                    f.write(pdf.read())
                log, files = call_backend(client, "ocr", gradio_file(tmp), int(dpi), int(batch_pages), dest3)
                st.text_area("Log", value=log or "", height=220)
                present_files(files, host_root)
            except Exception as e:
                st.error(str(e))
    st.markdown("---")
