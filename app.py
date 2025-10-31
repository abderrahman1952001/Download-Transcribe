# Streamlit client (FINAL) — matches final backend
import os, time
from urllib.parse import urlsplit
from gradio_client import Client, file as gradio_file
import streamlit as st

st.set_page_config(page_title="Media Toolkit Client", layout="wide")

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

def present_files(files):
    if not files:
        st.info("No files.")
        return
    for p in files:
        if isinstance(p, dict) and "url" in p:
            url = p["url"]; name = p.get("name") or os.path.basename(url)
            st.write(f"• [{name}]({url})")
        elif isinstance(p, str) and p.startswith("http"):
            name = os.path.basename(p.split("?")[0])
            st.write(f"• [{name}]({p})")
        else:
            st.code(str(p))

st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Colab URL (https://xxxx.gradio.live)", value="")

client = None
if backend_url.strip():
    try:
        client = ensure_client(backend_url)
        st.sidebar.success("Connected.")
    except Exception as e:
        st.sidebar.error(str(e))

st.title("Media Toolkit — Client (Final)")

tab1, tab2, tab3 = st.tabs(["Download","Transcribe","OCR"])

with tab1:
    st.subheader("Download")
    url = st.text_input("URL or ID", key="dl_url")
    processing = st.selectbox("Processing", ["None","Remove music (voice-only)"], index=0)
    out_choice = st.selectbox("Output", ["Video","Audio only"], index=0)
    quality = st.selectbox("Quality (Video only)", ["Best","1080p","720p","480p","360p"], index=0)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], index=0)
    if st.button("Download", type="primary", disabled=(client is None)):
        if not url.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "download", url, processing, out_choice, quality, dest_choice)
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))

with tab2:
    st.subheader("Transcribe (single video)")
    url2 = st.text_input("URL or ID", key="tr_url")
    lang = st.text_input("Language", value="ar")
    trans = st.checkbox("Translate to English", value=False)
    save_raw = st.checkbox("Save raw TXT", value=False)
    dest2 = st.selectbox("Destination", ["Here", "Drive"], index=0, key="tr_dest")
    if st.button("Transcribe", type="primary", disabled=(client is None)):
        if not url2.strip():
            st.warning("Paste a URL/ID first.")
        else:
            try:
                log, files = call_backend(client, "transcribe", url2, lang, bool(trans), bool(save_raw), dest2)
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))

with tab3:
    st.subheader("OCR")
    pdf = st.file_uploader("PDF file", type=["pdf"])
    dpi = st.slider("DPI", 150, 400, 300, 50)
    batch_pages = st.slider("Batch pages (affects RAM; for huge PDFs)", 10, 120, 40, 10)
    dest3 = st.selectbox("Destination", ["Here", "Drive"], index=0, key="ocr_dest")
    if st.button("Run OCR", type="primary", disabled=(client is None)):
        if not pdf:
            st.warning("Upload a PDF first.")
        else:
            try:
                tmp = f"_upload_{int(time.time())}.pdf"
                with open(tmp, "wb") as f: f.write(pdf.read())
                log, files = call_backend(client, "ocr", gradio_file(tmp), int(dpi), int(batch_pages), dest3)
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))
