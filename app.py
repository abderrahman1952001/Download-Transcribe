# streamlit run app.py
import os
import streamlit as st
from gradio_client import Client

st.set_page_config(page_title="Media Toolkit UI", page_icon="🎛️", layout="centered")
st.title("🎛️ Media Toolkit UI (Remote Colab Backend)")

st.markdown("""
Paste the **Colab backend URL** (the `https://...gradio.live` link that Colab printed).
Keep the Colab notebook open so the backend stays alive.
""")

backend_url = st.text_input(
    "Colab backend URL",
    placeholder="https://xxxxxxxxxxxxxxxx.gradio.live",
    help="Paste the public URL printed by the Colab backend cell."
)

@st.cache_resource(show_spinner=False)
def get_client(url: str):
    return Client(url)

def render_files(files):
    if not files:
        st.info("No files returned.")
        return
    st.success("Files ready:")
    for f in files:
        # gradio_client typically downloads returned files locally and returns local paths
        if isinstance(f, str) and os.path.isfile(f):
            fname = os.path.basename(f)
            with open(f, "rb") as fp:
                st.download_button(
                    label=f"⬇️ Download {fname}",
                    data=fp,
                    file_name=fname
                )
        else:
            # Fallback: show strings/URLs if backend returned non-local paths
            st.write("•", str(f))

tab_dl, tab_tr = st.tabs(["Download", "Transcribe"])

with tab_dl:
    url_or_id = st.text_input("Video URL or YouTube ID", placeholder="UF6qqLf_ZNA or https://…")
    processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"])
    output_kind = st.selectbox("Output", ["Video (best available)", "Audio only (best available)"])
    destination = st.selectbox("Destination", ["Here", "Drive"], help="Drive saves to /MyDrive/Colab_Media in your Google Drive")

    if st.button("Run Download"):
        if not backend_url or not url_or_id:
            st.warning("Enter backend URL and a video ID/URL.")
        else:
            try:
                client = get_client(backend_url)
                with st.spinner("Running on Colab backend…"):
                    log, files = client.predict(
                        url_or_id, processing, output_kind, destination,
                        api_name="/download"
                    )
                st.text_area("Log", value=log, height=220)
                render_files(files)
            except Exception as e:
                st.error(f"Backend error: {e}")

with tab_tr:
    url2 = st.text_input("Video URL or YouTube ID (for transcription)", placeholder="UF6qqLf_ZNA or https://…")
    language = st.text_input("Language (ISO code or 'auto')", value="ar")
    translate = st.checkbox("Translate to English", value=False)
    save_raw_txt = st.checkbox("Also save raw TXT (debug)", value=False)
    destination2 = st.selectbox("Destination", ["Here", "Drive"], key="dest2")

    if st.button("Run Transcription"):
        if not backend_url or not url2:
            st.warning("Enter backend URL and a video ID/URL.")
        else:
            try:
                client = get_client(backend_url)
                with st.spinner("Transcribing on Colab backend…"):
                    log, files = client.predict(
                        url2, language, translate, save_raw_txt, destination2,
                        api_name="/transcribe"
                    )
                st.text_area("Log", value=log, height=220)
                render_files(files)
            except Exception as e:
                st.error(f"Backend error: {e}")
