# app.py — Streamlit client (robust api_name resolution)
import io
import os
import time
import requests
import streamlit as st
from urllib.parse import urlsplit
from gradio_client import Client, file as gradio_file

st.set_page_config(page_title="Media Toolkit", layout="wide")

def ensure_client(url: str) -> Client:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Backend URL is empty.")
    p = urlsplit(url)
    base = f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else url
    return Client(base)

def present_files(paths):
    if not paths:
        st.info("No files returned.")
        return
    for p in paths:
        if isinstance(p, dict) and "url" in p:
            url = p["url"]; name = p.get("name") or os.path.basename(url)
            st.write(f"• [{name}]({url})")
        elif isinstance(p, str) and p.startswith("http"):
            name = os.path.basename(p.split("?")[0])
            st.write(f"• [{name}]({p})")
            try:
                with st.spinner("Fetching bytes…"):
                    r = requests.get(p, stream=True, timeout=120)
                    r.raise_for_status()
                    data = r.content if len(r.content) < 40_000_000 else None
                if data:
                    st.download_button("Download via Streamlit", data=data, file_name=name, key=f"dl_{name}")
                else:
                    st.caption("Large file: use the link above.")
            except Exception:
                pass
        else:
            st.code(str(p))

def call_backend(client: Client, api: str, *args):
    # Tries without and with leading slash for maximum compatibility
    names = [api.lstrip("/"), "/" + api.lstrip("/")]
    last_exc = None
    for n in names:
        try:
            return client.predict(*args, api_name=n)
        except Exception as e:
            last_exc = e
    # Raise the last error so the user can see it
    raise last_exc

st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Paste your Colab public URL (https://xxxx.gradio.live)", value="")
connected = False
client = None
if backend_url.strip():
    try:
        client = ensure_client(backend_url)
        connected = True
        st.sidebar.success("Connected ✅")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.title("🎛️ Media Toolkit — Client")

tab_dl, tab_tr, tab_ocr = st.tabs(["Download", "Transcribe", "OCR"])

with tab_dl:
    st.subheader("Download (YouTube/video/audio)")
    url_or_id = st.text_input("URL or ID", placeholder="YouTube URL or ID…")
    processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"])
    output_kind = st.selectbox("Output", ["Video (best available)", "Audio only (best available)"])
    dest_choice = st.selectbox("Destination", ["Here", "Drive"])

    if st.button("Run download", type="primary", disabled=not connected):
        if not connected:
            st.warning("Paste a backend URL first.")
        elif not url_or_id.strip():
            st.warning("Enter a URL or ID.")
        else:
            try:
                with st.spinner("Calling backend…"):
                    log, files = call_backend(
                        client, "download",
                        url_or_id.strip(),
                        processing,
                        output_kind,
                        dest_choice,
                    )
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))

with tab_tr:
    st.subheader("Transcribe (Whisper large-v3, chunked)")
    url_or_id_tr = st.text_input("URL or ID (audio/video)", placeholder="YouTube URL or ID…", key="tr_url")
    language = st.text_input("Language code (e.g., ar, en). Use 'ar' or leave blank for auto.", value="ar")
    translate = st.checkbox("Translate to English", value=False)
    save_raw_txt = st.checkbox("Also save raw TXT", value=False)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], key="tr_dest")

    if st.button("Run transcribe", type="primary", disabled=not connected):
        if not connected:
            st.warning("Paste a backend URL first.")
        elif not url_or_id_tr.strip():
            st.warning("Enter a URL or ID.")
        else:
            try:
                with st.spinner("Calling backend…"):
                    log, files = call_backend(
                        client, "transcribe",
                        url_or_id_tr.strip(),
                        language.strip(),
                        bool(translate),
                        bool(save_raw_txt),
                        dest_choice,
                    )
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))

with tab_ocr:
    st.subheader("OCR (Arabic PDF → polished DOCX)")
    pdf = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    dpi = st.slider("DPI", min_value=150, max_value=400, value=300, step=50)
    batch_pages = st.slider("Batch pages", min_value=10, max_value=120, value=40, step=10)
    fix_punct = st.checkbox("Fix Arabic punctuation & paragraphing", value=True)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], key="ocr_dest")

    if st.button("Run OCR", type="primary", disabled=not connected):
        if not connected:
            st.warning("Paste a backend URL first.")
        elif not pdf:
            st.warning("Upload a PDF first.")
        else:
            try:
                with st.spinner("Uploading & calling backend…"):
                    tmp_path = os.path.join(os.getcwd(), f"_upload_{int(time.time())}.pdf")
                    with open(tmp_path, "wb") as f:
                        f.write(pdf.read())
                    log, files = call_backend(
                        client, "ocr",
                        gradio_file(tmp_path),
                        int(dpi),
                        int(batch_pages),
                        bool(fix_punct),
                        dest_choice,
                    )
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))
