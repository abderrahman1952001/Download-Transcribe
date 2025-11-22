
# Colab Backend (FINAL-4)
import os, sys, subprocess, re, glob, shutil, time, uuid, urllib.parse
from typing import List, Tuple, Optional, Dict

def run(cmd, capture=True):
    if capture:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return (p.returncode or 0), p.stdout
    else:
        p = subprocess.run(cmd)
        return (p.returncode or 0), ""

def _ensure(pkgs: List[str]):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

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

def looks_like_playlist(url: str) -> bool:
    if not url: return False
    try:
        q = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(q.query)
        if "list" in qs: return True
        if "/playlist" in q.path: return True
    except Exception:
        pass
    return False

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

def is_audio(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in (".mp3",".m4a",".wav",".flac",".ogg",".opus",".aac")
def is_video(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in (".mp4",".mkv",".webm",".mov",".avi")

def list_new(out_dir: str, before: set) -> List[str]:
    return [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f not in before]

def ensure_drive_once(summary: List[str]) -> bool:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        summary.append("Drive: not Colab → skip mount.")
        return False
    if os.path.ismount('/content/drive'):
        return True
    try:
        summary.append("Mounting Drive…")
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        summary.append(f"Drive mount failed: {e}")
    return os.path.ismount('/content/drive')

def copy_to_drive(paths: List[str], summary: List[str]) -> List[str]:
    if not paths:
        return []
    mounted = ensure_drive_once(summary)
    if not mounted:
        summary.append("Drive not mounted → skip copy.")
        return []
    root = "/content/drive/MyDrive/MediaToolkit"
    os.makedirs(root, exist_ok=True)
    copied = []
    for p in paths:
        try:
            tgt = os.path.join(root, os.path.basename(p))
            shutil.copy2(p, tgt)
            if os.path.exists(tgt) and os.path.getsize(tgt) > 0:
                copied.append(tgt)
        except Exception as e:
            summary.append(f"Copy fail: {os.path.basename(p)} → {e}")
    if copied:
        summary.append(f"Drive copies: {len(copied)} file(s).")
    return copied

def build_outtmpl(out_dir: str, url: str) -> str:
    if looks_like_playlist(url):
        return os.path.join(out_dir, "%(playlist_index)03d - %(title).200s.%(ext)s")
    else:
        return os.path.join(out_dir, "%(title).200s.%(ext)s")

def download_media(url: str, out_dir: str, audio_only=False, quality="Best"):
    os.makedirs(out_dir, exist_ok=True)
    outtmpl = build_outtmpl(out_dir, url)
    ydl_opts = dict(
        outtmpl=outtmpl, quiet=True, no_warnings=True, ignoreerrors=True,
        noprogress=True, continuedl=True, retries=3, fragment_retries=5,
        format=quality_to_format(quality, audio_only=audio_only),
        postprocessors=[]
    )
    if audio_only:
        ydl_opts["postprocessors"] = [{"key":"FFmpegExtractAudio","preferredcodec":"mp3","preferredquality":"192"}]
    files_before = set(os.listdir(out_dir))
    summary = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            summary.append("yt-dlp: done.")
    except Exception as e:
        summary.append(f"yt-dlp error: {e!r}")
    new_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f not in files_before]
    new_files.sort()
    return summary, new_files

def demucs_vocals(audio_path: str, out_dir: str, summary: list, model_name: str = "mdx_extra_q"):
    try:
        sep_root = os.path.join(out_dir, "demucs", os.path.splitext(os.path.basename(audio_path))[0] + "-" + uuid.uuid4().hex[:6])
        os.makedirs(sep_root, exist_ok=True)
        code, out = run([sys.executable,"-m","demucs","-n",model_name,"--two-stems","vocals","-o", sep_root, audio_path])
        if code != 0:
            summary.append(f"Demucs failed ({model_name}).")
            return None
        vocals = glob.glob(os.path.join(sep_root,"**","vocals.wav"), recursive=True)
        if not vocals:
            summary.append("Demucs: no vocals.wav.")
            return None
        voc = vocals[0]
        out_wav = os.path.join(out_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".voice.wav")
        code, _ = run(["ffmpeg","-y","-i",voc,"-ac","1","-ar","16000", out_wav])
        if code == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav)>0:
            return out_wav
        summary.append("ffmpeg resample failed.")
        return None
    except Exception as e:
        summary.append(f"Demucs exception: {e}")
        return None

def merge_video_with_audio(video_path: str, audio_path: str, out_dir: str, summary: list):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_vid = os.path.join(out_dir, base + ".voice.mp4")
    code, _ = run(["ffmpeg","-y","-i",video_path,"-i",audio_path,
                   "-c:v","copy","-map","0:v:0","-map","1:a:0","-shortest", out_vid])
    if code == 0 and os.path.exists(out_vid) and os.path.getsize(out_vid)>0:
        return out_vid
    summary.append("ffmpeg mux failed.")
    return None

def _extract_audio_from_video(video_path: str, summary: list) -> str|None:
    out_mp3 = os.path.splitext(video_path)[0] + ".mp3"
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
        return out_mp3
    code, _ = run(["ffmpeg","-y","-i",video_path,"-vn","-acodec","libmp3lame","-q:a","2", out_mp3])
    if code == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3)>0:
        return out_mp3
    summary.append(f"ffmpeg extract audio failed for {os.path.basename(video_path)}")
    return None

def _group_by_basename(paths: list[str]) -> dict[str, dict[str, list[str]]]:
    M = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        t = "audio" if is_audio(p) else ("video" if is_video(p) else "other")
        if t == "other": continue
        M.setdefault(base, {"audio": [], "video": []})
        M[base][t].append(p)
    return M

def api_download(url_or_id: str, processing: str, output_kind: str, quality: str, save_to_drive: bool):
    t0 = time.time()
    url = normalize_url(url_or_id)
    if not url: return "Paste a URL/ID.", []
    out_dir = "/content/outputs/download"; os.makedirs(out_dir, exist_ok=True)
    audio_only = output_kind.startswith("Audio")
    summary = []

    s1, paths = download_media(url, out_dir, audio_only=audio_only, quality=quality)
    summary += s1

    final_paths: list[str] = []

    if processing.startswith("Remove"):
        grouped = _group_by_basename(paths)
        for base, kinds in grouped.items():
            made_voice = None
            aud = kinds["audio"][0] if kinds["audio"] else None
            vid = kinds["video"][0] if kinds["video"] else None
            if not aud and vid:
                aud = _extract_audio_from_video(vid, summary)
            if aud:
                made_voice = demucs_vocals(aud, out_dir, summary, model_name="mdx_extra_q")
            if audio_only:
                if made_voice: final_paths.append(made_voice)
            else:
                if not vid:
                    # fetch corresponding video (rare case: audio-only initial download)
                    s2, p2 = download_media(url, out_dir, audio_only=False, quality=quality)
                    summary += s2
                    for p in p2:
                        if os.path.splitext(os.path.basename(p))[0] == base and is_video(p):
                            vid = p; break
                if made_voice and vid:
                    merged = merge_video_with_audio(vid, made_voice, out_dir, summary)
                    if merged: final_paths.append(merged)
        if not final_paths:
            summary.append("Voice-only requested but no processed output produced.")
    else:
        if audio_only:
            grouped = _group_by_basename(paths)
            for base, kinds in grouped.items():
                if kinds["audio"]:
                    final_paths.extend(kinds["audio"])
                elif kinds["video"]:
                    aud = _extract_audio_from_video(kinds["video"][0], summary)
                    if aud: final_paths.append(aud)
        else:
            final_paths = [p for p in paths if is_video(p)]

    if save_to_drive and final_paths:
        copy_to_drive(final_paths, summary)

    dt = time.time() - t0
    summary.insert(0, f"Download ✓  ({len(final_paths)} file(s), {dt:.1f}s)")
    return "\n".join(summary), final_paths

# Transcription
def split_audio_chunks(audio_path: str, chunk_minutes=20, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(audio_path)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    chunk_dir = os.path.join(out_dir, base + "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    pattern = os.path.join(chunk_dir, "chunk_%03d.wav")
    src16 = os.path.join(chunk_dir, "src16.wav")
    run(["ffmpeg","-y","-i",audio_path,"-ac","1","-ar","16000", src16])
    seg_time = str(int(chunk_minutes*60))
    run(["ffmpeg","-y","-i",src16,"-f","segment","-segment_time",seg_time,"-c","copy", pattern])
    chunks = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.wav")))
    return chunks

def write_docx(paras, docx_path: str):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.shared import OxmlElement
    doc = Document()
    for p in paras:
        pr = doc.add_paragraph(p)
        try:
            pr.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pPr = pr._element.get_or_add_pPr()
            bidi = OxmlElement('w:bidi')
            pPr.append(bidi)
        except Exception:
            pass
    doc.save(docx_path)

def assemble_paragraphs(text: str):
    text = re.sub(r"\s+([،؛:,.!?])", r"\1", text)
    text = re.sub(r"([،؛:,.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    sentences = re.split(r"(?<=[.؟!])\s+", text)
    paras, cur = [], ""
    for s in sentences:
        cand = (cur + " " + s).strip()
        if len(cand) < 300:
            cur = cand; continue
        if len(cand) > 1000:
            paras.append(cur.strip()); cur = s
        else:
            cur = cand
    if cur.strip(): paras.append(cur.strip())
    return paras

def _prepare_audio_for_transcribe(url_or_id: str, summary: list):
    url = normalize_url(url_or_id)
    if not url:
        summary.append("Empty URL/ID."); return None
    tmp_dir = "/content/outputs/transcribe"; os.makedirs(tmp_dir, exist_ok=True)
    s1, paths1 = download_media(url, tmp_dir, audio_only=True, quality="Best")
    summary += s1
    for p in paths1:
        if is_audio(p): return p
    s2, paths2 = download_media(url, tmp_dir, audio_only=False, quality="Best")
    summary += s2
    for p in paths2:
        if is_video(p):
            out_mp3 = os.path.splitext(p)[0] + ".mp3"
            code, _ = run(["ffmpeg","-y","-i",p,"-vn","-acodec","libmp3lame","-q:a","2", out_mp3])
            if code == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3)>0:
                return out_mp3
    summary.append("No audio extracted from source.")
    return None

def api_transcribe(url_or_id: str, language: str, save_raw_txt: bool, save_to_drive: bool):
    t0 = time.time()
    summary = []
    audio_path = _prepare_audio_for_transcribe(url_or_id, summary)
    if not audio_path:
        return "Download/extract audio failed.", []

    chunks = split_audio_chunks(audio_path, chunk_minutes=20)
    if not chunks:
        chunk_dir = os.path.join(os.path.dirname(audio_path), "single_chunk")
        os.makedirs(chunk_dir, exist_ok=True)
        single = os.path.join(chunk_dir, "chunk_000.wav")
        run(["ffmpeg","-y","-i",audio_path,"-ac","1","-ar","16000", single])
        chunks = [single]
    summary.append(f"Transcribe: {len(chunks)} chunk(s).")

    from faster_whisper import WhisperModel
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    lang = None if (language or "").lower() in ("auto","") else language
    all_text = []
    for i, wav in enumerate(chunks, 1):
        segments, _ = model.transcribe(wav, task="transcribe", language=lang,
                                       vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        lines = [s.text.strip() for s in segments if (s.text or "").strip()]
        all_text.append(" ".join(lines))
        summary.append(f"Chunk {i}/{len(chunks)} ✓")

    text = " ".join(all_text)
    paras = assemble_paragraphs(text)

    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = "/content/outputs/transcribe"; os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, base + ".docx")
    write_docx(paras, docx_path)

    outputs = [docx_path]
    if save_raw_txt:
        raw = os.path.join(out_dir, base + ".txt")
        with open(raw, "w", encoding="utf-8") as f: f.write(text)
        outputs.append(raw)

    if save_to_drive:
        copy_to_drive(outputs, summary)

    dt = time.time() - t0
    summary.insert(0, f"Transcribe ✓  ({len(chunks)} chunk(s), {dt:.1f}s)")
    return "\n".join(summary), outputs

# OCR
def ocr_pdf_to_docx(src_pdf: str, out_dir: str, dpi=300, batch_pages=40, summary=None):
    if summary is None: summary = []
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_pdf))[0]

    text = ""
    if have_pdftotext():
        code, out = run(["pdftotext", "-enc", "UTF-8", src_pdf, "-"])
        if code == 0 and out.strip():
            text = out
            summary.append("pdftotext path used.")

    if not text.strip():
        import numpy as np, cv2, easyocr
        from pdf2image import convert_from_path
        total_pages = None
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(src_pdf, userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0)) or None
        except Exception:
            total_pages = None

        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        reader = easyocr.Reader(['ar','en'], gpu=use_gpu)
        lines = []

        def process_pages(pages):
            for pil in pages:
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
                thr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                result = reader.readtext(thr, detail=0, paragraph=True)
                lines.extend([x.strip() for x in result if x and x.strip()])

        if total_pages:
            first = 1
            while first <= total_pages:
                last = min(first + int(batch_pages) - 1, total_pages)
                pages = convert_from_path(src_pdf, dpi=dpi, first_page=first, last_page=last)
                process_pages(pages)
                summary.append(f"OCR pages {first}-{last} ✓")
                first = last + 1
        else:
            pages = convert_from_path(src_pdf, dpi=dpi)
            process_pages(pages)
            summary.append("OCR all pages ✓")

        text = "\n".join(lines)

    paras = assemble_paragraphs(text)
    out_docx = os.path.join(out_dir, base + ".docx")
    write_docx(paras, out_docx)
    return [out_docx]

def api_ocr(pdf_file, dpi: int, batch_pages: int, save_to_drive: bool):
    t0 = time.time()
    src = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    out_dir = "/content/outputs/ocr"; os.makedirs(out_dir, exist_ok=True)
    summary = []
    out_files = ocr_pdf_to_docx(src, out_dir, dpi=int(dpi), batch_pages=int(batch_pages), summary=summary)
    if save_to_drive:
        copy_to_drive(out_files, summary)
    dt = time.time() - t0
    summary.insert(0, f"OCR ✓  (1 file, {dt:.1f}s)")
    return "\n".join(summary), out_files

# ---- TabbedInterface only ----
download_iface = gr.Interface(fn=api_download,
                              inputs=[gr.Textbox(label="URL or ID"),
                                      gr.Dropdown(["None","Remove music (voice-only)"], label="Processing"),
                                      gr.Dropdown(["Video","Audio only"], label="Output"),
                                      gr.Dropdown(["Best","1080p","720p","480p","360p"], label="Quality (Video only)"),
                                      gr.Checkbox(label="Save to Drive")],
                              outputs=[gr.Textbox(label="Summary", lines=8), gr.Files(label="Outputs")],
                              allow_flagging="never", api_name="download")

transcribe_iface = gr.Interface(fn=api_transcribe,
                                inputs=[gr.Textbox(label="URL or ID"),
                                        gr.Textbox(value="ar", label="Language (e.g., ar or auto)"),
                                        gr.Checkbox(label="Save raw TXT"),
                                        gr.Checkbox(label="Save to Drive")],
                                outputs=[gr.Textbox(label="Summary", lines=8), gr.Files(label="Outputs")],
                                allow_flagging="never", api_name="transcribe")

ocr_iface = gr.Interface(fn=api_ocr,
                         inputs=[gr.File(label="PDF file", file_types=[".pdf"]),
                                 gr.Slider(150,400,300,50, label="DPI"),
                                 gr.Slider(10,120,40,10, label="Batch pages (RAM control)"),
                                 gr.Checkbox(label="Save to Drive")],
                         outputs=[gr.Textbox(label="Summary", lines=8), gr.Files(label="Outputs")],
                         allow_flagging="never", api_name="ocr")

app = gr.TabbedInterface([download_iface, transcribe_iface, ocr_iface], tab_names=["download","transcribe","ocr"])

if __name__ == "__main__":
    app.queue().launch(share=True, inline=False)


# Minimal keep-alive so Colab doesn't mark the runtime idle
import time, threading
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
