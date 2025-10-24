# app.py
import io
import os
import time
import requests
import streamlit as st
from urllib.parse import urlparse
from gradio_client import Client, file as gradio_file

st.set_page_config(page_title="Media Toolkit", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
def ensure_client(url: str) -> Client:
    # Normalize: accept raw host or full URL
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("Backend URL is empty.")
    # Allow both https://xxxxx.gradio.live and https://xxxxx.gradio.live/download etc.
    # Client needs the root base URL only
    return Client(url)

def is_http_link(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def present_files(files):
    if not files:
        st.info("No files returned.")
        return
    for p in files:
        name = os.path.basename(p)
        if is_http_link(p):
            # Show link; offer download if you want
            with st.container():
                st.markdown(f"**Result:** [{name}]({p})")
                # Optional download via Streamlit (careful: large files suck through the app)
                try:
                    with st.spinner("Fetching bytes…"):
                        r = requests.get(p, stream=True, timeout=60)
                        r.raise_for_status()
                        data = r.content if len(r.content) < 40_000_000 else None  # cap ~40MB
                    if data:
                        st.download_button("Download via Streamlit", data=data, file_name=name)
                    else:
                        st.caption("Large file: use the link above to download directly.")
                except Exception:
                    st.caption("Direct link provided above; app-side download skipped.")
        else:
            # Non-HTTP path (e.g., Colab local or Drive path string)
            st.code(p)

# ------------------------------
# Sidebar (backend + destination)
# ------------------------------
st.sidebar.header("Backend")
backend_url = st.sidebar.text_input(
    "Colab backend URL (from gradio.live)",
    value=os.getenv("BACKEND_URL", ""),
    placeholder="https://xxxx.gradio.live"
)
if "backend_url" not in st.session_state:
    st.session_state.backend_url = backend_url
if backend_url and backend_url != st.session_state.backend_url:
    st.session_state.backend_url = backend_url

dest_choice = st.sidebar.selectbox("Where to save on backend?", ["Here", "Drive"], index=0)
st.sidebar.caption("“Drive” copies outputs to /MyDrive/Colab_Media on your Google Drive.")

connected = False
client = None
if st.session_state.backend_url:
    try:
        client = ensure_client(st.session_state.backend_url)
        connected = True
        st.sidebar.success("Connected ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {e}")

st.title("🎛️ Media Toolkit")
st.caption("Download • Transcribe • OCR (Arabic-tuned)")

# ------------------------------
# Tabs
# ------------------------------
tab_dl, tab_tr, tab_ocr = st.tabs(["Download", "Transcribe", "OCR"])

# ---- Download Tab ----
with tab_dl:
    st.subheader("Download (Video/Audio, optional music removal)")
    url_or_id = st.text_input("Video URL or YouTube ID", placeholder="UF6qqLf_ZNA or https://…")
    c1, c2, c3 = st.columns(3)
    with c1:
        processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"], index=0)
    with c2:
        output_kind = st.selectbox("Output", ["Video (best available)", "Audio only (best available)"], index=0)
    with c3:
        st.write("")  # spacing
        run_dl = st.button("Run download", type="primary", use_container_width=True)

    if run_dl:
        if not connected:
            st.error("Connect to backend first (left sidebar).")
        elif not url_or_id.strip():
            st.warning("Provide a URL or ID.")
        else:
            with st.spinner("Processing on backend…"):
                try:
                    # /download returns: [log_str, list_of_files]
                    log, files = client.predict(
                        url_or_id,
                        processing,
                        output_kind,
                        dest_choice,
                        api_name="/download",
                    )
                    st.text_area("Log", value=log, height=220)
                    present_files(files)
                except Exception as e:
                    st.error(str(e))

# ---- Transcribe Tab ----
with tab_tr:
    st.subheader("Transcribe (Arabic lecture-tuned)")
    url_or_id_tr = st.text_input("Video URL or YouTube ID (for audio extraction)", key="tr_url")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        language = st.text_input("Language", value="ar")
    with c2:
        translate = st.checkbox("Translate to English", value=False)
    with c3:
        save_raw_txt = st.checkbox("Also save raw TXT", value=False)
    with c4:
        st.write("")  # spacing
        run_tr = st.button("Run transcription", type="primary", use_container_width=True)

    if run_tr:
        if not connected:
            st.error("Connect to backend first (left sidebar).")
        elif not url_or_id_tr.strip():
            st.warning("Provide a URL or ID.")
        else:
            with st.spinner("Transcribing on backend…"):
                try:
                    log, files = client.predict(
                        url_or_id_tr,
                        language,
                        bool(translate),
                        bool(save_raw_txt),
                        dest_choice,
                        api_name="/transcribe",
                    )
                    st.text_area("Log", value=log, height=220)
                    present_files(files)
                except Exception as e:
                    st.error(str(e))

# ---- OCR Tab ----
with tab_ocr:
    st.subheader("OCR (Arabic PDF → polished DOCX)")
    pdf = st.file_uploader("Upload PDF (Arabic)", type=["pdf"])
    c1, c2, c3 = st.columns(3)
    with c1:
        dpi = st.slider("DPI", min_value=150, max_value=400, step=50, value=300)
    with c2:
        batch_pages = st.slider("Batch size (pages)", min_value=10, max_value=120, step=10, value=40)
    with c3:
        fix_punct = st.checkbox("Fix punctuation & paragraphing", value=True)
    run_ocr = st.button("Run OCR", type="primary")

    if run_ocr:
        if not connected:
            st.error("Connect to backend first (left sidebar).")
        elif pdf is None:
            st.warning("Upload a PDF.")
        else:
            with st.spinner("OCR running on backend…"):
                try:
                    # Streamlit gives us a BytesIO; write to temp so gradio_client can upload
                    tmp_path = os.path.join("/tmp", pdf.name)
                    with open(tmp_path, "wb") as f:
                        f.write(pdf.read())

                    log, files = client.predict(
                        gradio_file(tmp_path),
                        int(dpi),
                        int(batch_pages),
                        bool(fix_punct),
                        dest_choice,
                        api_name="/ocr",
                    )
                    st.text_area("Log", value=log, height=220)
                    present_files(files)
                except Exception as e:
                    st.error(str(e))
