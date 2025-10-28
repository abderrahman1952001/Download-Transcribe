
import os
import io
import time
import json
import traceback
import streamlit as st

# Try to ensure gradio_client is available (works on local runs; some hosted envs require requirements.txt)
try:
    from gradio_client import Client
except Exception:
    try:
        import sys, subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio_client>=0.6.3"], check=False)
        from gradio_client import Client
    except Exception as e:
        st.error(f"Failed to import or install gradio_client: {e}")
        st.stop()

st.set_page_config(page_title="Downloader / Transcriber", page_icon="🎧", layout="centered")
st.title("🎧 Downloader / Transcriber — Streamlit UI")
st.caption("Connect to your running **Colab Gradio backend** and call its named API endpoints.")

# --------------- State ---------------
if "client" not in st.session_state:
    st.session_state.client = None
if "api_info" not in st.session_state:
    st.session_state.api_info = ""

# --------------- Connect panel ---------------
with st.container():
    backend_url = st.text_input("Gradio backend URL", placeholder="https://xxxxxxxxxxxx.gradio.live", help="Paste the exact URL printed by your Colab backend.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("Connect", use_container_width=True):
            if not backend_url.strip():
                st.error("Paste the backend URL first.")
            else:
                try:
                    c = Client(backend_url.strip())
                    info = c.view_api()
                    st.session_state.client = c
                    st.session_state.api_info = info
                    st.success("Connected.")
                except Exception as e:
                    st.session_state.client = None
                    st.error(f"Failed to connect: {e}")
    with colB:
        if st.button("Ping", use_container_width=True, disabled=st.session_state.client is None):
            try:
                out = st.session_state.client.predict(api_name="/ping")
                st.toast(f"Ping: {out}", icon="✅")
            except Exception as e:
                st.error(f"Ping failed: {e}")
    with colC:
        if st.button("View API", use_container_width=True, disabled=st.session_state.client is None):
            if st.session_state.api_info:
                st.code(st.session_state.api_info, language="json")
            else:
                try:
                    st.session_state.api_info = st.session_state.client.view_api()
                    st.code(st.session_state.api_info, language="json")
                except Exception as e:
                    st.error(f"view_api() failed: {e}")

st.markdown("---")

if st.session_state.client is None:
    st.info("Connect to the backend to enable actions.")
    st.stop()

tabs = st.tabs(["⬇️ Download", "🗣️ Transcribe", "📄 OCR PDF"])

# --------------- Download tab ---------------
with tabs[0]:
    st.subheader("Download video / audio (supports playlists)")
    url = st.text_input("YouTube URL or ID (playlist or single)")
    processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"])
    output_kind = st.selectbox("Output kind", ["Video", "Audio"])
    destination = st.selectbox("Destination", ["Local", "Drive"], help="Drive requires Colab / Google Drive.")
    if st.button("Run download", type="primary"):
        try:
            paths = st.session_state.client.predict(
                url_or_id=url,
                processing=processing,
                output_kind=output_kind,
                destination=destination,
                api_name="/download"
            )
            st.success("Done.")
            st.write(paths)
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.exception(e)

# --------------- Transcribe tab ---------------
with tabs[1]:
    st.subheader("Transcribe a video (audio extracted automatically)")
    url_t = st.text_input("YouTube URL or ID")
    lang = st.text_input("Language code (e.g., ar, en, fr). Leave blank or 'auto' to autodetect.", value="ar")
    translate = st.checkbox("Translate to English (Whisper translate task)", value=False)
    save_raw = st.checkbox("Also save raw .txt", value=True)
    destination_t = st.selectbox("Destination", ["Local", "Drive"], key="dest_t")
    if st.button("Run transcribe", type="primary"):
        try:
            outputs = st.session_state.client.predict(
                url_or_id=url_t,
                language=lang,
                translate=translate,
                save_raw_txt=save_raw,
                destination=destination_t,
                api_name="/transcribe"
            )
            st.success("Done.")
            st.write(outputs)
        except Exception as e:
            st.error(f"Transcribe failed: {e}")
            st.exception(e)

# --------------- OCR tab ---------------
with tabs[2]:
    st.subheader("OCR a PDF (URL) — Arabic-friendly")
    pdf_url = st.text_input("PDF URL (publicly accessible). If you have a local file, upload it somewhere or place it on Drive and paste a 'uc?export=download' link.")
    dpi = st.number_input("Render DPI (images)", min_value=150, max_value=400, value=300, step=25)
    batch_pages = st.number_input("Pages per batch (OCR)", min_value=1, max_value=20, value=5, step=1,
                                  help="Helps with memory in Colab.")
    fix_punct = st.checkbox("Polish Arabic punctuation", value=True)
    destination_o = st.selectbox("Destination", ["Local", "Drive"], key="dest_o")
    if st.button("Run OCR", type="primary"):
        try:
            outputs = st.session_state.client.predict(
                pdf_file=pdf_url,
                dpi=int(dpi),
                batch_pages=int(batch_pages),
                fix_punct=bool(fix_punct),
                destination=destination_o,
                api_name="/ocr"
            )
            st.success("Done.")
            st.write(outputs)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.exception(e)
