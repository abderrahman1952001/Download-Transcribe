# Streamlit client v3 — robust api_name + endpoint discovery
import os, time, json, requests, streamlit as st
from urllib.parse import urlsplit
from gradio_client import Client, file as gradio_file

st.set_page_config(page_title="Media Toolkit", layout="wide")

def base_root(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    p = urlsplit(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}"
    return url

def ensure_client(url: str) -> Client:
    return Client(base_root(url))

def discover_endpoints(root: str):
    try:
        r = requests.get(root + "/config", timeout=10)
        r.raise_for_status()
        data = r.json()
        names = []
        for dep in data.get("dependencies", []):
            n = dep.get("api_name")
            if n: names.append(n.lstrip("/"))
        return list(dict.fromkeys(names))
    except Exception:
        return []

def call_backend(client: Client, desired: str, *args):
    candidates = [desired, desired.lstrip("/"), "/" + desired.lstrip("/")]
    tried = []
    for n in candidates:
        try:
            return client.predict(*args, api_name=n)
        except Exception as e:
            tried.append((n, str(e)))
    raise RuntimeError("Tried: " + ", ".join([x for x,_ in tried]) + "\nLast: " + tried[-1][1])

def present_files(paths):
    if not paths:
        st.info("No files returned."); return
    for p in paths:
        if isinstance(p, dict) and "url" in p:
            url = p["url"]; name = p.get("name") or os.path.basename(url)
            st.write(f"• [{name}]({url})")
        elif isinstance(p, str) and p.startswith("http"):
            name = os.path.basename(p.split("?")[0])
            st.write(f"• [{name}]({p})")
        else:
            st.code(str(p))

st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Paste your Colab URL (https://xxxx.gradio.live)", value="")
client = None
endpoints = []
if backend_url.strip():
    try:
        client = ensure_client(backend_url)
        st.sidebar.success("Connected ✅")
        root = base_root(backend_url)
        endpoints = discover_endpoints(root)
        if endpoints:
            st.sidebar.caption("Endpoints: " + ", ".join(endpoints))
        else:
            st.sidebar.caption("Could not auto-discover endpoints.")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.title("🎛️ Media Toolkit — Client v3")

tab_dl, tab_tr, tab_ocr = st.tabs(["Download", "Transcribe", "OCR"])

with tab_dl:
    st.subheader("Download")
    url_or_id = st.text_input("URL or ID")
    processing = st.selectbox("Processing", ["None", "Remove music (voice-only)"])
    output_kind = st.selectbox("Output", ["Video (best available)", "Audio only (best available)"])
    dest_choice = st.selectbox("Destination", ["Here", "Drive"])
    if st.button("Run download", type="primary", disabled=(client is None)):
        try:
            log, files = call_backend(client, "download", url_or_id.strip(), processing, output_kind, dest_choice)
            st.text_area("Log", value=log, height=220)
            present_files(files)
        except Exception as e:
            st.error(str(e))

with tab_tr:
    st.subheader("Transcribe")
    url_or_id_tr = st.text_input("URL or ID (audio/video)", key="tr")
    language = st.text_input("Language code (e.g., ar, en).", value="ar")
    translate = st.checkbox("Translate to English", value=False)
    save_raw_txt = st.checkbox("Also save raw TXT", value=False)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], key="tr_dest")
    if st.button("Run transcribe", type="primary", disabled=(client is None)):
        try:
            log, files = call_backend(client, "transcribe", url_or_id_tr.strip(), language.strip(), bool(translate), bool(save_raw_txt), dest_choice)
            st.text_area("Log", value=log, height=220)
            present_files(files)
        except Exception as e:
            st.error(str(e))

with tab_ocr:
    st.subheader("OCR (PDF → DOCX)")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    dpi = st.slider("DPI", 150, 400, 300, 50)
    batch_pages = st.slider("Batch pages", 10, 120, 40, 10)
    dest_choice = st.selectbox("Destination", ["Here", "Drive"], key="ocr_dest")
    if st.button("Run OCR", type="primary", disabled=(client is None)):
        if not pdf:
            st.warning("Upload a PDF first.")
        else:
            try:
                tmp = f"_upload_{int(time.time())}.pdf"
                with open(tmp, "wb") as f: f.write(pdf.read())
                log, files = call_backend(client, "ocr", gradio_file(tmp), int(dpi), int(batch_pages), dest_choice)
                st.text_area("Log", value=log, height=220)
                present_files(files)
            except Exception as e:
                st.error(str(e))
