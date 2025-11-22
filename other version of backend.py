# Colab Backend (FINAL) — Download / Transcribe / OCR
# Features:
# - Download (single or playlist): quality select for Video, Audio-only, optional "Remove music (voice-only)".
#   * Demucs runs per-item with explicit output folders to avoid collisions.
#   * If Video+voice-only: we re-mux voice-only audio back over the original video.
# - Transcribe: single video/ID only; always polish (dedupe, punctuation, paragraphs). No 'polish' arg exposed.
# - OCR: always deskew + Arabic punctuation normalization; now truly batches large PDFs using pdfinfo -> first/last page windows.
#
import os, sys, subprocess, re, glob, shutil
from typing import List, Tuple, Optional

def run(cmd: list, timeout: Optional[int] = None) -> Tuple[int, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    try:
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if line:
                out.append(line)
    finally:
        try:
            p.wait(timeout=timeout)
        except Exception:
            p.kill()
    return p.returncode or 0, "".join(out)

def _ensure(pkgs: List[str]):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

# ---------- Deps (Colab-friendly) ----------
REQS = [
    "yt-dlp>=2024.8.6",
    "gradio>=4.18.0",
    "faster-whisper>=1.0.3",
    "demucs>=4.0.1",
    "python-docx>=1.1.2",
    "easyocr>=1.7.1",
    "pdf2image>=1.17.0",
    "pillow>=10.3.0",
    "opencv-python-headless>=4.9.0.80",
]
try:
    import yt_dlp, gradio as gr  # noqa
except Exception:
    _ensure(REQS)
    import yt_dlp, gradio as gr  # noqa

def have_pdftotext():
    try:
        return subprocess.call(["pdftotext", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False

# ---------- Helpers ----------
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
_YT_PLAYLIST_ID_RE = re.compile(r"^(PL|UU|LL|OL)[A-Za-z0-9_-]{10,}$")

def normalize_url(s: str) -> str:
    s = (s or "").strip()
    if not s: return s
    if _YT_ID_RE.match(s):
        return f"https://www.youtube.com/watch?v={s}"
    if _YT_PLAYLIST_ID_RE.match(s):
        return f"https://www.youtube.com/playlist?list={s}"
    if "://" not in s:
        return "https://" + s
    return s

def quality_to_format(quality: str, audio_only: bool) -> str:
    if audio_only:
        return "bestaudio/best"
    q = (quality or "Best").lower()
    if q == "1080p":
        return "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    if q == "720p":
        return "bestvideo[height<=720]+bestaudio/best[height<=720]"
    if q == "480p":
        return "bestvideo[height<=480]+bestaudio/best[height<=480]"
    if q == "360p":
        return "bestvideo[height<=360]+bestaudio/best[height<=360]"
    return "bestvideo+bestaudio/best"

def list_new(out_dir: str, before: set) -> List[str]:
    return [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f not in before]

def is_audio(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in (".mp3",".m4a",".wav",".flac",".ogg")
def is_video(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in (".mp4",".mkv",".webm",".mov",".avi")

def ensure_drive_once():
    try:
        from google.colab import drive  # type: ignore
        if not os.path.ismount('/content/drive'):
            drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass

def copy_to_destination(paths: List[str], destination: str) -> List[str]:
    if destination == "Drive":
        ensure_drive_once()
        root = "/content/drive/MyDrive/MediaToolkit"
        os.makedirs(root, exist_ok=True)
        out = []
        for p in paths:
            tgt = os.path.join(root, os.path.basename(p))
            try:
                shutil.copy2(p, tgt); out.append(tgt)
            except Exception:
                out.append(p)
        return out
    return paths

# ---------- Download core (with Demucs fixes) ----------
def download_media(url: str, out_dir: str, audio_only=False, voice_only=False, quality="Best"):
    os.makedirs(out_dir, exist_ok=True)
    base_tmpl = os.path.join(out_dir, "%(title).200s-%(id)s.%(ext)s")
    ydl_opts = dict(
        outtmpl=base_tmpl, quiet=True, no_warnings=True, ignoreerrors=True,
        noprogress=True, continuedl=True, retries=3, fragment_retries=5,
        format=quality_to_format(quality, audio_only=audio_only),
        postprocessors=[]
    )
    if audio_only:
        ydl_opts["postprocessors"] = [{"key":"FFmpegExtractAudio","preferredcodec":"mp3","preferredquality":"192"}]

    files_before = set(os.listdir(out_dir))
    log = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            log.append("yt-dlp finished.")
    except Exception as e:
        log.append(f"yt-dlp error: {e!r}")
    new_files = list_new(out_dir, files_before)

    # If voice_only requested, run Demucs for ALL audio targets (playlist-safe)
    if voice_only and new_files:
        import uuid
        voice_wavs = []
        audio_targets = [p for p in new_files if is_audio(p)]
        video_targets = [p for p in new_files if is_video(p)]
        if not audio_targets and video_targets:
            for v in video_targets:
                out_mp3 = os.path.splitext(v)[0] + ".mp3"
                code, out = run(["ffmpeg","-y","-i",v,"-vn","-acodec","libmp3lame","-q:a","2", out_mp3])
                log.append(out);
                if code == 0: audio_targets.append(out_mp3)

        for aud in audio_targets:
            sep_root = os.path.join(out_dir, "demucs", os.path.splitext(os.path.basename(aud))[0] + "-" + uuid.uuid4().hex[:6])
            os.makedirs(sep_root, exist_ok=True)
            code, out = run([sys.executable,"-m","demucs","--two-stems","vocals","-o", sep_root, aud])
            log.append(out)
            stems = glob.glob(os.path.join(sep_root,"**","vocals.wav"), recursive=True)
            if stems:
                voc = stems[0]
                out_wav = os.path.join(out_dir, os.path.splitext(os.path.basename(aud))[0] + ".voice.wav")
                code, out = run(["ffmpeg","-y","-i",voc,"-ac","1","-ar","16000", out_wav])
                log.append(out)
                if code == 0:
                    voice_wavs.append((aud, out_wav))
                    new_files.append(out_wav)

        # If VIDEO output, re-mux voice track over original video(s)
        if not audio_only and video_targets and voice_wavs:
            for v in video_targets:
                base = os.path.splitext(os.path.basename(v))[0]
                match = None
                for (aud, vw) in voice_wavs:
                    if os.path.basename(vw).startswith(base):
                        match = vw; break
                if match is None and voice_wavs:
                    match = voice_wavs[0][1]
                if match:
                    out_vid = os.path.join(out_dir, base + ".voice.mp4")
                    code, out = run(["ffmpeg","-y","-i",v,"-i",match,
                                     "-c:v","copy","-map","0:v:0","-map","1:a:0","-shortest", out_vid])
                    log.append(out)
                    if code == 0:
                        new_files.append(out_vid)

    return "\n".join(log), new_files

# ---------- Transcribe (always polish) ----------
def _prepare_audio_for_transcribe(url_or_id: str) -> Tuple[str, str]:
    url = normalize_url(url_or_id)
    if not url: raise ValueError("Empty URL/ID.")
    tmp_dir = "/content/outputs/transcribe"; os.makedirs(tmp_dir, exist_ok=True)
    log, paths = download_media(url, tmp_dir, audio_only=True, voice_only=False, quality="Best")
    for p in paths:
        if is_audio(p):
            return p, log
    for p in paths:
        if is_video(p):
            out_mp3 = os.path.splitext(p)[0] + ".mp3"
            code, out = run(["ffmpeg","-y","-i",p,"-vn","-acodec","libmp3lame","-q:a","2", out_mp3])
            return out_mp3, (log + "\n" + out)
    raise RuntimeError("No audio extracted from the provided source.")

def dedupe_lines(lines: List[str], threshold=92.0) -> List[str]:
    try:
        from rapidfuzz import fuzz  # optional
    except Exception:
        return lines
    out = []
    for line in lines:
        if not out:
            out.append(line); continue
        if all(fuzz.token_sort_ratio(line, x) < threshold for x in out):
            out.append(line)
    return out

def assemble_paragraphs(lines: List[str], min_len=250, max_len=900) -> List[str]:
    lines = dedupe_lines(lines)
    paras, cur = [], ""
    for line in lines:
        cand = (cur + " " + line).strip()
        if len(cand) < min_len:
            cur = cand; continue
        if len(cand) > max_len:
            paras.append(cur.strip()); cur = line
        else:
            cur = cand
    if cur.strip(): paras.append(cur.strip())
    return paras

def xml_safe(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)

def write_docx(paras: List[str], docx_path: str):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.shared import OxmlElement
    doc = Document()
    for p in paras:
        p = xml_safe(p)
        pr = doc.add_paragraph(p)
        try:
            pr.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            # set RTL via XML flag (w:bidi)
            pPr = pr._element.get_or_add_pPr()
            bidi = OxmlElement('w:bidi')
            pPr.append(bidi)
        except Exception:
            pass
    doc.save(docx_path)

def transcribe_once(audio_path: str, out_dir: str, language="ar", translate=False, save_raw_txt=False) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    from faster_whisper import WhisperModel
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    task = "translate" if translate else "transcribe"
    lang = None if (language or "").lower() in ("auto","") else language

    # 16k mono resample
    wav = os.path.join(out_dir, "input_16k.wav")
    run(["ffmpeg","-y","-i",audio_path,"-ac","1","-ar","16000", wav])

    segments, _ = model.transcribe(wav, task=task, language=lang, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    lines = [s.text.strip() for s in segments if (s.text or "").strip()]

    # Arabic punctuation & spacing
    text = " ".join(lines)
    text = re.sub(r"\s+([،؛:,.!?])", r"\1", text)
    text = re.sub(r"([،؛:,.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    paras = assemble_paragraphs(re.split(r"(?<=[.؟!])\s+", text))

    outputs = []
    docx_path = os.path.join(out_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".docx")
    write_docx(paras, docx_path); outputs.append(docx_path)
    if save_raw_txt:
        raw_path = os.path.join(out_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".txt")
        with open(raw_path, "w", encoding="utf-8") as f: f.write(text)
        outputs.append(raw_path)
    return outputs

# ---------- OCR (true batching) ----------
def ocr_pdf_to_docx(src_pdf: str, out_dir: str, dpi=300, batch_pages=40) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_pdf))[0]
    txt_fast = ""

    # Fast path for text PDFs
    if have_pdftotext():
        code, out = run(["pdftotext", "-enc", "UTF-8", src_pdf, "-"])
        if code == 0 and out.strip():
            txt_fast = out

    if txt_fast.strip():
        text = txt_fast
    else:
        import numpy as np, cv2, easyocr
        from pdf2image import convert_from_path
        # Try to get page count to enable batching
        total_pages = None
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(src_pdf, userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0)) or None
        except Exception:
            total_pages = None

        reader = easyocr.Reader(['ar','en'], gpu=True)
        lines = []

        if total_pages:
            first = 1
            while first <= total_pages:
                last = min(first + int(batch_pages) - 1, total_pages)
                try:
                    pages = convert_from_path(src_pdf, dpi=dpi, first_page=first, last_page=last)
                except Exception:
                    # fallback: whole PDF
                    pages = convert_from_path(src_pdf, dpi=dpi)
                    first, last, total_pages = 1, len(pages), len(pages)
                for pil in pages:
                    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
                    thr = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, 41, -0.2) if hasattr(cv2, "ximgproc") else cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                    result = reader.readtext(thr, detail=0, paragraph=True)
                    lines.extend([x.strip() for x in result if x and x.strip()])
                first = last + 1
        else:
            pages = convert_from_path(src_pdf, dpi=dpi)
            for pil in pages:
                import numpy as np, cv2
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
                thr = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, 41, -0.2) if hasattr(cv2, "ximgproc") else cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                result = reader.readtext(thr, detail=0, paragraph=True)
                lines.extend([x.strip() for x in result if x and x.strip()])

        text = "\\n".join(lines)

    # Arabic punctuation normalization
    text = re.sub(r"\\s+([،؛:,.!?])", r"\\1", text)
    text = re.sub(r"([،؛:,.!?])([^\\s])", r"\\1 \\2", text)
    text = re.sub(r"\\s{2,}", " ", text).strip()

    paras = assemble_paragraphs(re.split(r"(?<=[.؟!])\\s+", text))
    out_docx = os.path.join(out_dir, base + ".docx")
    write_docx(paras, out_docx)
    return [out_docx]

# ---------- API wrappers ----------
def api_download(url_or_id: str, processing: str, output_kind: str, quality: str, destination: str):
    url = normalize_url(url_or_id)
    if not url: return "Empty URL/ID.", []
    out_dir = "/content/outputs/download"; os.makedirs(out_dir, exist_ok=True)
    audio_only = output_kind.startswith("Audio")
    voice_only = processing.startswith("Remove")
    log, paths = download_media(url, out_dir, audio_only=audio_only, voice_only=voice_only, quality=quality)
    return log, copy_to_destination(paths, destination)

def api_transcribe(url_or_id: str, language: str, translate: bool, save_raw_txt: bool, destination: str):
    try:
        audio_path, dl_log = _prepare_audio_for_transcribe(url_or_id)
    except Exception as e:
        return f"Download failed: {e}", []
    out_dir = "/content/outputs/transcribe"; os.makedirs(out_dir, exist_ok=True)
    out_files = transcribe_once(audio_path, out_dir, language=language or "ar",
                                translate=bool(translate), save_raw_txt=bool(save_raw_txt))
    return dl_log, copy_to_destination(out_files, destination)

def api_ocr(pdf_file, dpi: int, batch_pages: int, destination: str):
    src = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    out_dir = "/content/outputs/ocr"; os.makedirs(out_dir, exist_ok=True)
    out_files = ocr_pdf_to_docx(src, out_dir, dpi=int(dpi), batch_pages=int(batch_pages))
    return f"OCR complete: {os.path.basename(src)}", copy_to_destination(out_files, destination)

# ---------- Gradio App ----------
with gr.Blocks() as demo:
    gr.Markdown("## Media Toolkit — Final Backend")

    with gr.Tab("Download"):
        url = gr.Textbox(label="url_or_id")
        processing = gr.Dropdown(choices=["None","Remove music (voice-only)"], value="None", label="processing")
        output_kind = gr.Dropdown(choices=["Video","Audio only"], value="Video", label="output_kind")
        quality = gr.Dropdown(choices=["Best","1080p","720p","480p","360p"], value="Best", label="quality (Video only)")
        destination = gr.Dropdown(choices=["Here","Drive"], value="Here", label="destination")
        btn = gr.Button("Download", variant="primary")
        log = gr.Textbox(label="log")
        files = gr.Files(label="files")
        btn.click(api_download, inputs=[url, processing, output_kind, quality, destination], outputs=[log, files])

    with gr.Tab("Transcribe"):
        url2 = gr.Textbox(label="url_or_id")
        lang = gr.Textbox(value="ar", label="language")
        trans = gr.Checkbox(value=False, label="translate to English")
        save_raw = gr.Checkbox(value=False, label="save_raw_txt")
        dest2 = gr.Dropdown(choices=["Here","Drive"], value="Here", label="destination")
        btn2 = gr.Button("Transcribe", variant="primary")
        log2 = gr.Textbox(label="log")
        files2 = gr.Files(label="files")
        btn2.click(api_transcribe, inputs=[url2, lang, trans, save_raw, dest2], outputs=[log2, files2])

    with gr.Tab("OCR"):
        pdf = gr.File(label="PDF file", file_types=[".pdf"])
        dpi = gr.Slider(150, 400, value=300, step=50, label="DPI")
        batch = gr.Slider(10, 120, value=40, step=10, label="Batch pages (affects RAM; for huge PDFs)")
        dest3 = gr.Dropdown(choices=["Here","Drive"], value="Here", label="destination")
        btn3 = gr.Button("OCR", variant="primary")
        log3 = gr.Textbox(label="log")
        files3 = gr.Files(label="files")
        btn3.click(api_ocr, inputs=[pdf, dpi, batch, dest3], outputs=[log3, files3])

# Expose stable endpoints
download_iface = gr.Interface(fn=api_download,
                              inputs=[gr.Textbox(), gr.Dropdown(["None","Remove music (voice-only)"]),
                                      gr.Dropdown(["Video","Audio only"]), gr.Dropdown(["Best","1080p","720p","480p","360p"]),
                                      gr.Dropdown(["Here","Drive"])],
                              outputs=[gr.Textbox(), gr.Files()], allow_flagging="never", api_name="download")
transcribe_iface = gr.Interface(fn=api_transcribe,
                                inputs=[gr.Textbox(), gr.Textbox(value="ar"), gr.Checkbox(), gr.Checkbox(), gr.Dropdown(["Here","Drive"])],
                                outputs=[gr.Textbox(), gr.Files()], allow_flagging="never", api_name="transcribe")
ocr_iface = gr.Interface(fn=api_ocr,
                         inputs=[gr.File(), gr.Slider(150,400,300,50), gr.Slider(10,120,40,10), gr.Dropdown(["Here","Drive"])],
                         outputs=[gr.Textbox(), gr.Files()], allow_flagging="never", api_name="ocr")

app = gr.TabbedInterface([download_iface, transcribe_iface, ocr_iface], tab_names=["download","transcribe","ocr"])

if __name__ == "__main__":
  app.queue().launch(
        share=True,          # force a gradio.live URL
        inline=False,        # don't render a widget inside Colab (fixes the JS/cookies error)
    )
