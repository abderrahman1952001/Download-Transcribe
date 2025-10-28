import streamlit as st
from gradio_client import Client
import json
import time

st.set_page_config(page_title="🎧 Downloader / Transcriber — Streamlit UI")

st.title("🎧 Downloader / Transcriber — Streamlit UI")
st.caption("Connect to your running Colab Gradio backend and call its named API endpoints.")

backend_url = st.text_input("Gradio backend URL", placeholder="https://xxxxx.gradio.live/")
connect = st.button("Connect / Refresh endpoints")

if "client" not in st.session_state:
    st.session_state.client = None
if "apis" not in st.session_state:
    st.session_state.apis = {}

def fetch_api_map(client: Client):
    """
    Returns: dict name->fn_index from client.view_api_info()
    """
    info = client.view_api_info()  # dict with 'named_endpoints'
    apis = {}
    try:
        named = info.get("named_endpoints") or {}
        for name, meta in named.items():
            # Normalize both "download" and "/download" to the same key for matching
            clean = name.strip()
            apis[clean] = meta.get("fn_index")
            # also register variant without leading slash
            if clean.startswith("/"):
                apis[clean[1:]] = meta.get("fn_index")
            else:
                apis["/" + clean] = meta.get("fn_index")
    except Exception:
        pass
    return apis

def require_connection():
    if not st.session_state.client:
        st.error("Connect to the backend first.")
        st.stop()

if connect and backend_url.strip():
    try:
        st.session_state.client = Client(backend_url.strip())
        st.session_state.apis = fetch_api_map(st.session_state.client)
        st.success(f"Connected. Found endpoints: {', '.join(sorted(st.session_state.apis.keys())) or 'none'}")
    except Exception as e:
        st.session_state.client = None
        st.session_state.apis = {}
        st.error(f"Connection failed: {e}")

st.write("---")
tab_dl, tab_tr, tab_ocr = st.tabs(["⬇️ Download", "🗣️ Transcribe", "📄 OCR PDF"])

def call_named(api_preferred_list, **kwargs):
    """
    Try preferred names, then fallback by fuzzy lookup.
    """
    client = st.session_state.client
    apis = st.session_state.apis
    # First pass: try exact preferred names
    for name in api_preferred_list:
        if name in apis:
            return client.predict(api_name=name, **kwargs)
    # Second pass: heuristic pick (e.g., find 'download' substring)
    for key in apis.keys():
        if api_preferred_list[0].strip("/").lower() in key.strip("/").lower():
            return client.predict(api_name=key, **kwargs)
    # Last resort: call first function index
    if apis:
        some_name = next(iter(apis.keys()))
        return client.predict(api_name=some_name, **kwargs)
    raise RuntimeError("No named endpoints found on backend.")

with tab_dl:
    st.subheader("Download video / audio (supports playlists)")
    url = st.text_input("YouTube URL or ID (playlist or single)")
    processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"], index=0)
    kind = st.selectbox("Output kind", ["Video","Audio"], index=0)
    dest = st.selectbox("Destination", ["Local","Drive"], index=0)
    if st.button("Start download"):
        require_connection()
        try:
            paths = call_named(
                ["/download","download"],
                url_or_id=url,
                processing=processing,
                output_kind=kind,
                destination=dest,
            )
            st.success("Done.")
            st.json(paths)
        except Exception as e:
            st.error(f"Download failed: {e}")

with tab_tr:
    st.subheader("Transcribe by URL/ID")
    url_t = st.text_input("YouTube URL or ID", key="tr_url")
    lang = st.text_input("Language ('' or 'auto' to let model decide)", value="ar")
    translate = st.checkbox("Translate to English", value=False)
    rawtxt = st.checkbox("Also save raw .txt", value=False)
    dest_t = st.selectbox("Destination", ["Local","Drive"], index=0, key="dest_tr")
    if st.button("Start transcription"):
        require_connection()
        try:
            paths = call_named(
                ["/transcribe","transcribe"],
                url_or_id=url_t,
                language=lang,
                translate=translate,
                save_raw_txt=rawtxt,
                destination=dest_t,
            )
            st.success("Done.")
            st.json(paths)
        except Exception as e:
            st.error(f"Transcription failed: {e}")

with tab_ocr:
    st.subheader("OCR a PDF (Arabic-aware, smart paragraphs)")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    dpi = st.slider("DPI", 100, 500, 300, step=25)
    fix = st.checkbox("Polish Arabic punctuation/paragraphs", value=True)
    dest_o = st.selectbox("Destination", ["Local","Drive"], index=0, key="dest_ocr")
    if st.button("Start OCR"):
        require_connection()
        if not uploaded:
            st.warning("Upload a PDF first.")
        else:
            try:
                # gradio_client expects ('pdf_file',) style tuple for file
                paths = call_named(
                    ["/ocr","ocr"],
                    pdf_file=uploaded,
                    dpi=int(dpi),
                    batch_pages=False,
                    fix_punct=bool(fix),
                    destination=dest_o,
                )
                st.success("Done.")
                st.json(paths)
            except Exception as e:
                st.error(f"OCR failed: {e}")
