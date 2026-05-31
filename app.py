import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Fix gradio_client 1.3.0 + pydantic v2: _json_schema_to_python_type() crashes when
# additionalProperties is True (boolean). Wrap the function to return "Any" in that case.
try:
    import gradio_client.utils as _gcu
    _orig_j2p = _gcu._json_schema_to_python_type

    def _safe_j2p(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _orig_j2p(schema, defs)
        except Exception:
            return "Any"

    _gcu._json_schema_to_python_type = _safe_j2p

    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "unknown"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type
except Exception:
    pass

import sys
import threading
import time
import queue
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from utils import (
    detect_device, get_env_info, get_video_info, format_video_info_md,
    open_in_finder, notify_macos, find_model_weights, download_weights,
)

# ---------------------------------------------------------------------------
# CSS — system-aware light / dark theme, SF Pro, no hardcoded colours
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ───────────────────────── Design tokens ───────────────────────── */
:root {
  --bg:            #f0f0f2;
  --surface:       #ffffff;
  --surface-2:     #f6f6f8;
  --inset:         #ececf0;
  --text:          #1d1d1f;
  --text-2:        #6e6e73;
  --text-3:        #8e8e93;
  --accent:        #0071e3;
  --accent-hover:  #0077ed;
  --accent-soft:   rgba(0,113,227,0.10);
  --on-accent:     #ffffff;
  --success:       #1aa54a;
  --warning:       #c77700;
  --error:         #e0352b;
  --border:        rgba(0,0,0,0.10);
  --border-strong: rgba(0,0,0,0.16);
  --ring:          rgba(0,113,227,0.35);
  --shadow:        0 1px 2px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.05);
  --radius:        14px;
  --radius-sm:     9px;
  --radius-xs:     7px;
  --r-track:       #d9d9de;
}

/* Dark theme — Gradio toggles `.dark` on <body> from the system color scheme.
   (We can't use `@media (prefers-color-scheme: dark) { :root {...} }` because
   Gradio's CSS scoper rewrites selectors nested in @media into
   `... .contain :root`, which never matches the document root.) */
.dark {
  --bg:            #161618;
  --surface:       #1f1f22;
  --surface-2:     #29292d;
  --inset:         #161618;
  --text:          #f5f5f7;
  --text-2:        #a1a1a8;
  --text-3:        #79797f;
  --accent:        #0a84ff;
  --accent-hover:  #3b9dff;
  --accent-soft:   rgba(10,132,255,0.16);
  --on-accent:     #ffffff;
  --success:       #30d158;
  --warning:       #ffd60a;
  --error:         #ff5247;
  --border:        rgba(255,255,255,0.10);
  --border-strong: rgba(255,255,255,0.16);
  --ring:          rgba(10,132,255,0.45);
  --shadow:        0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.35);
  --r-track:       #46464b;
}

/* ───────────────────────── Base ───────────────────────── */
* {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
               'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}

/* Make the whole viewport one uniform colour: the centered 900px container sits
   on top of `gradio-app`, which Gradio paints white by default — without this it
   shows as side-bands around the column. */
body, gradio-app {
  background: var(--bg) !important;
}
.gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  max-width: 900px !important;
  margin: 0 auto !important;
  padding: 48px 28px !important;
  font-size: 15px !important;
  line-height: 1.5 !important;
  -webkit-font-smoothing: antialiased;
}
.gradio-container *,
.gradio-container .prose,
.gradio-container p,
.gradio-container span,
.gradio-container li { color: var(--text); }

/* ───────────────────────── App header ───────────────────────── */
/* The first markdown title ("# RVM — Détourage Vidéo") gets a gradient logo. */
#app-title h1 {
  display: flex !important; align-items: center; gap: 13px;
  font-size: 27px !important; font-weight: 700 !important;
  letter-spacing: -0.02em !important; margin: 0 0 4px !important;
}
#app-title h1::before {
  content: ""; width: 38px; height: 38px; border-radius: 10px; flex: none;
  display: inline-block;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 4 L20 4 L20 16 L4 16 Z M2 20 L22 20 M9 9.5 L9 11 M15 9.5 L15 11 M9 13 Q12 15 15 13'/%3E%3C/svg%3E") center / 21px no-repeat,
    linear-gradient(150deg, var(--accent), #8a5cff);
  box-shadow: 0 3px 10px var(--accent-soft);
}
#app-subtitle p { color: var(--text-2) !important; font-size: 14px !important; margin: 0 0 30px !important; }

/* ───────────────────────── Cards ───────────────────────── */
.card {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  padding: 22px 24px 24px !important;
  margin-bottom: 18px !important;
}
/* Neutralise the default grey block backgrounds Gradio wraps around content,
   so text/markdown sits flush on the white (or dark) card surface. */
.card .styler,
.card .form,
.card > div > .block,
.card fieldset.block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ───────────────────────── Section headings ───────────────────────── */
.section-title h2 {
  font-size: 17px !important; font-weight: 650 !important;
  letter-spacing: -0.01em !important; margin: 0 0 18px !important;
  color: var(--text) !important;
}

/* ───────────────────────── Status glyphs ───────────────────────── */
.status-ok   { color: var(--success); font-weight: 600; }
.status-warn { color: var(--warning); font-weight: 600; }
.status-err  { color: var(--text-2);  font-weight: 600; }

/* ───────────────────────── Buttons ───────────────────────── */
.gradio-container button {
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  transition: background .15s, border-color .15s, opacity .15s;
}

/* NOTE: Gradio applies elem_classes directly on the <button>. Selectors below
   target both that case and a possible nested <button> wrapper for robustness. */

/* Primary action */
.btn-primary, .btn-primary button {
  background: var(--accent) !important;
  color: var(--on-accent) !important;
  border: 1px solid transparent !important;
}
.btn-primary:hover, .btn-primary button:hover { background: var(--accent-hover) !important; }

/* Danger / cancel — outline style */
.btn-danger, .btn-danger button {
  background: transparent !important;
  color: var(--error) !important;
  border: 1px solid color-mix(in oklch, var(--error), transparent 60%) !important;
}
.btn-danger:hover, .btn-danger button:hover {
  background: color-mix(in oklch, var(--error), transparent 90%) !important;
}
.btn-danger:disabled, .btn-danger button:disabled { opacity: 0.45 !important; cursor: not-allowed; }

/* Secondary (download / finder) */
.ico-download, .ico-download button, .ico-folder, .ico-folder button {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
}
.ico-download:hover, .ico-download button:hover,
.ico-folder:hover, .ico-folder button:hover { background: var(--inset) !important; }

/* ───────────────────────── Button icons (CSS mask) ───────────────────────── */
.ico-play::before, .ico-play button::before,
.ico-stop::before, .ico-stop button::before,
.ico-download::before, .ico-download button::before,
.ico-folder::before, .ico-folder button::before,
.ico-prev::before, .ico-prev button::before,
.ico-next::before, .ico-next button::before {
  content: ""; display: inline-block; width: 16px; height: 16px;
  margin-right: 8px; vertical-align: -3px; flex: none;
  background: currentColor;
  -webkit-mask: var(--svg) center / contain no-repeat;
          mask: var(--svg) center / contain no-repeat;
}
.ico-play::before, .ico-play button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Cpath d='M7 5v14l11-7z'/%3E%3C/svg%3E");
}
.ico-stop::before, .ico-stop button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='6' y='6' width='12' height='12' rx='2'/%3E%3C/svg%3E");
}
.ico-download::before, .ico-download button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3v12m0 0l-4-4m4 4l4-4M5 21h14'/%3E%3C/svg%3E");
}
.ico-folder::before, .ico-folder button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z'/%3E%3C/svg%3E");
}

/* ───────────────────────── Preview nav buttons (round) ───────────────────────── */
.nav-btn, .nav-btn button {
  width: 38px !important; height: 38px !important; min-width: 38px !important;
  border-radius: 50% !important; padding: 0 !important;
  background: var(--surface-2) !important; color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  display: grid !important; place-items: center !important;
}
.nav-btn:hover, .nav-btn button:hover { background: var(--inset) !important; }
.ico-prev::before, .ico-prev button::before,
.ico-next::before, .ico-next button::before { margin-right: 0; width: 16px; height: 16px; }
.ico-prev::before, .ico-prev button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5l-7 7 7 7'/%3E%3C/svg%3E");
}
.ico-next::before, .ico-next button::before {
  --svg: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 5l7 7-7 7'/%3E%3C/svg%3E");
}

/* ───────────────────────── Inputs ───────────────────────── */
.gradio-container input[type=text],
.gradio-container input[type=number],
.gradio-container textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
.gradio-container input[type=text]:focus,
.gradio-container input[type=number]:focus,
.gradio-container textarea:focus {
  outline: none !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--ring) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: var(--text-3) !important; }

/* ───────────────────────── Sliders ───────────────────────── */
.gradio-container input[type=range] {
  -webkit-appearance: none; appearance: none; width: 100%; height: 6px;
  background: var(--r-track) !important; border-radius: 999px; outline: none;
}
.gradio-container input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%;
  background: #fff; border: 0.5px solid rgba(0,0,0,0.12);
  box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer;
}
.gradio-container input[type=range]::-moz-range-thumb {
  width: 20px; height: 20px; border-radius: 50%; background: #fff;
  border: 0.5px solid rgba(0,0,0,0.12); box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer;
}

/* ───────────────────────── Radios / checkboxes ───────────────────────── */
.gradio-container .wrap label,
.gradio-container fieldset label {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
.gradio-container input[type=radio]:checked + span,
.gradio-container input[type=checkbox]:checked + span { color: var(--accent) !important; }

/* ───────────────────────── Log textarea ───────────────────────── */
.log-box textarea {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace !important;
  font-size: 12px !important;
  background: var(--inset) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

/* ───────────────────────── Labels ───────────────────────── */
.gradio-container label > span,
.gradio-container .gr-form > div > span {
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
"""

# ---------------------------------------------------------------------------
# Globals shared between UI and inference thread
# ---------------------------------------------------------------------------

_cancel_event = threading.Event()
_progress_queue: "queue.Queue[tuple]" = queue.Queue()
_current_output_dir: Optional[Path] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checker_preview(w: int = 400, h: int = 225) -> np.ndarray:
    cell = 20
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(0, h, cell):
        for col in range(0, w, cell):
            v = 192 if ((row // cell) + (col // cell)) % 2 == 0 else 128
            img[row:row+cell, col:col+cell] = v
    return img

BG_COLORS_UI = {
    "Noir":   (0, 0, 0),
    "Blanc":  (255, 255, 255),
    "Vert chroma": (0, 177, 64),
    "Damier": None,  # handled separately
}
BG_MAP = {
    "Noir":  "black",
    "Blanc": "white",
    "Vert chroma": "green",
    "Damier": "checker",
}

BACKBONE_MAP = {
    "Rapide — MobileNetV3": "mobilenetv3",
    "Qualité max — ResNet50": "resnet50",
}

RESOLUTION_MAP = {
    "Auto (recommandé)": None,
    "Qualité max — 0.5": 0.5,
    "Rapide — 0.25": 0.25,
    "Manuel": "manual",
}

OUTPUT_FORMAT_MAP = {
    "Vidéo (MP4)": "video",
    "Séquence PNG": "png",
    "Les deux": "both",
}

OUTPUT_NAMES = {
    "Composition finale":    "composite",
    "Alpha mask seul":       "alpha",
    "Foreground brut (RGB)": "foreground",
}


def _extract_preview_frames(video_path: str, bg_name: str) -> list:
    """Extract 4 preview frames at 10/25/50/75% of the video."""
    path = Path(video_path)
    if not path.exists():
        return [None, None, None, None]

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [int(n * p) for p in (0.10, 0.25, 0.50, 0.75)]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        else:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    cap.release()

    # Apply background tint for preview
    result = []
    for f in frames:
        if bg_name == "Damier":
            h, w = f.shape[:2]
            result.append(_make_checker_preview(w, h))
        else:
            color = BG_COLORS_UI.get(bg_name, (0, 0, 0))
            bg = np.full_like(f, color, dtype=np.uint8)
            result.append(bg)
    return result  # raw bg frames for "before" panel — source frames for source panel


def _source_frames(video_path: str) -> list:
    """Return 4 source frames."""
    if not video_path or not Path(video_path).exists():
        return [None] * 4
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [int(n * p) for p in (0.10, 0.25, 0.50, 0.75)]
    out = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ok, frame = cap.read()
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    return out


def _load_result_frames(output_dir: Path, stem: str, bg_name: str) -> list:
    """Load composite frames from the output directory for preview."""
    comp_video = output_dir / f"{stem}_composite.mp4"
    comp_dir   = output_dir / f"{stem}_composite_png"

    frames = []
    if comp_video.exists():
        cap = cv2.VideoCapture(str(comp_video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for p in (0.10, 0.25, 0.50, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * p))
            ok, frame = cap.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
        cap.release()
    elif comp_dir.exists():
        pngs = sorted(comp_dir.glob("*.png"))
        n = len(pngs)
        for p in (0.10, 0.25, 0.50, 0.75):
            idx = int(n * p)
            if idx < n:
                frames.append(np.array(Image.open(pngs[idx]).convert("RGB")))
            else:
                frames.append(None)
    return frames if frames else [None, None, None, None]


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="RVM — Détourage Vidéo") as demo:

        # State
        video_info_state   = gr.State({})
        src_frames_state   = gr.State([None]*4)
        result_frames_state = gr.State([None]*4)
        preview_idx_state  = gr.State(0)

        gr.Markdown("# RVM — Détourage Vidéo", elem_id="app-title")
        gr.Markdown("*Interface locale pour Robust Video Matting — macOS uniquement*", elem_id="app-subtitle")

        # ── Section 1 : Environnement ────────────────────────────────────

        with gr.Group(elem_classes=["card"]):
            gr.Markdown("## 1 · Environnement", elem_classes=["section-title"])
            env_md = gr.Markdown(get_env_info())

            with gr.Row():
                weights_status_md = gr.Markdown(_weights_status_text())
                with gr.Column(scale=0, min_width=220):
                    dl_backbone = gr.Radio(
                        ["MobileNetV3", "ResNet50"],
                        value="MobileNetV3",
                        label="Poids à télécharger",
                        visible=_any_weights_missing(),
                    )
                    dl_btn = gr.Button(
                        "Télécharger les poids",
                        visible=_any_weights_missing(),
                        variant="secondary",
                        elem_classes=["ico-download"],
                    )
            dl_progress = gr.Progress()
            dl_status   = gr.Markdown("")

        # ── Section 2 : Import vidéo ─────────────────────────────────────

        with gr.Group(elem_classes=["card"]):
            gr.Markdown("## 2 · Importer une vidéo", elem_classes=["section-title"])
            video_input = gr.File(
                label="Glissez votre vidéo ici ou cliquez pour parcourir",
                file_types=[".mp4", ".mov", ".mkv", ".avi"],
                type="filepath",
            )
            video_meta_md  = gr.Markdown("")
            video_thumb    = gr.Image(label="Première frame", visible=False, height=180)

        # ── Section 3 : Réglages ──────────────────────────────────────────

        with gr.Group(elem_classes=["card"]):
            gr.Markdown("## 3 · Réglages", elem_classes=["section-title"])

            with gr.Row():
                with gr.Column():
                    backbone_radio = gr.Radio(
                        list(BACKBONE_MAP.keys()),
                        value="Rapide — MobileNetV3",
                        label="Modèle",
                        info="MobileNetV3 : rapide, très bon pour la majorité des cas. ResNet50 : contours plus fins, plus lent.",
                    )

                with gr.Column():
                    resolution_radio = gr.Radio(
                        list(RESOLUTION_MAP.keys()),
                        value="Auto (recommandé)",
                        label="Résolution de traitement",
                        info="Auto cible 768px internes (meilleure qualité par défaut). Qualité max=0.5 pour des bords encore plus fins (plus lent).",
                    )
                    manual_ds = gr.Number(
                        value=0.25,
                        label="Ratio manuel (0.05 – 0.5)",
                        minimum=0.05,
                        maximum=0.5,
                        step=0.05,
                        visible=False,
                    )

            with gr.Row():
                with gr.Column():
                    seq_chunk_slider = gr.Slider(
                        minimum=4, maximum=24, value=12, step=1,
                        label="Parallélisme (seq_chunk)",
                        info="Plus élevé = plus rapide mais consomme plus de RAM. Sur Apple Silicon, 12–16 est optimal.",
                    )

                with gr.Column():
                    alpha_sharpness_slider = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.0, step=0.1,
                        label="Netteté des bords (gamma alpha)",
                        info="1.0 = neutre. >1 = contours plus tranchés, moins de halo. <1 = bords plus doux/flous.",
                    )

            with gr.Row():
                with gr.Column():
                    output_format_radio = gr.Radio(
                        list(OUTPUT_FORMAT_MAP.keys()),
                        value="Vidéo (MP4)",
                        label="Format de sortie",
                    )

            with gr.Row():
                with gr.Column():
                    outputs_check = gr.CheckboxGroup(
                        list(OUTPUT_NAMES.keys()),
                        value=["Composition finale"],
                        label="Sorties à générer",
                    )

                with gr.Column():
                    bg_radio = gr.Radio(
                        list(BG_COLORS_UI.keys()),
                        value="Noir",
                        label="Fond de prévisualisation",
                    )

            with gr.Row():
                output_dir_box = gr.Textbox(
                    label="Dossier de sortie",
                    placeholder="Laisser vide = Bureau + nom_vidéo_RVM_output",
                    scale=4,
                )

        # ── Section 4 : Lancement ────────────────────────────────────────

        with gr.Group(elem_classes=["card"]):
            gr.Markdown("## 4 · Lancement", elem_classes=["section-title"])

            with gr.Row():
                launch_btn = gr.Button(
                    "Lancer le détourage",
                    variant="primary",
                    scale=3,
                    elem_classes=["btn-primary", "ico-play"],
                )
                cancel_btn = gr.Button(
                    "Annuler",
                    variant="stop",
                    scale=1,
                    elem_classes=["btn-danger", "ico-stop"],
                )

            progress_bar = gr.Progress()
            progress_md  = gr.Markdown("")
            log_box      = gr.Textbox(
                label="Journal",
                lines=6,
                max_lines=12,
                interactive=False,
                elem_classes=["log-box"],
            )
            result_md    = gr.Markdown("")

            with gr.Row(visible=False) as finder_row:
                finder_btn = gr.Button("Ouvrir dans le Finder", variant="secondary", elem_classes=["ico-folder"])

        # ── Section 5 : Prévisualisation ──────────────────────────────────

        with gr.Group(elem_classes=["card"]):
            gr.Markdown("## 5 · Prévisualisation", elem_classes=["section-title"])
            gr.Markdown("*Naviguez entre 4 moments clés de votre vidéo (10%, 25%, 50%, 75%)*")

            with gr.Row():
                prev_btn = gr.Button("", scale=0, min_width=50, elem_classes=["nav-btn", "ico-prev"])
                with gr.Column(scale=2):
                    frame_label = gr.Markdown("**Frame 1/4** — 10%")
                next_btn = gr.Button("", scale=0, min_width=50, elem_classes=["nav-btn", "ico-next"])

            with gr.Row():
                source_img = gr.Image(label="Source", height=280, show_label=True)
                result_img = gr.Image(label="Résultat", height=280, show_label=True)

            preview_hint = gr.Markdown(
                "*Lancez le détourage pour voir le résultat côte à côte.*",
                visible=True,
            )

        # ──────────────────────────────────────────────────────────────────
        # Event handlers
        # ──────────────────────────────────────────────────────────────────

        # ── Video import ──
        def on_video_upload(filepath):
            if not filepath:
                return (
                    "",
                    gr.update(visible=False),
                    {},
                    [None]*4,
                )
            info = get_video_info(Path(filepath))
            meta_text = format_video_info_md(info)
            thumb = info.get("thumbnail")
            src_frames = _source_frames(filepath)
            # Suggest output dir on Desktop
            suggested_out = str(Path.home() / "Desktop" / (Path(filepath).stem + "_RVM_output"))
            return (
                meta_text,
                gr.update(value=thumb, visible=thumb is not None),
                info,
                src_frames,
            )

        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[video_meta_md, video_thumb, video_info_state, src_frames_state],
        )

        # Update output dir suggestion when video changes
        def suggest_output_dir(filepath):
            if not filepath:
                return ""
            return str(Path.home() / "Desktop" / (Path(filepath).stem + "_RVM_output"))

        video_input.change(
            fn=suggest_output_dir,
            inputs=[video_input],
            outputs=[output_dir_box],
        )

        # ── Resolution manual field ──
        def on_resolution_change(val):
            return gr.update(visible=(val == "Manuel"))

        resolution_radio.change(
            fn=on_resolution_change,
            inputs=[resolution_radio],
            outputs=[manual_ds],
        )

        # ── Download weights ──
        def on_download_weights(backbone_choice, progress=gr.Progress()):
            backbone = "mobilenetv3" if "Mobile" in backbone_choice else "resnet50"
            try:
                def cb(frac, msg):
                    progress(frac, desc=msg)
                download_weights(backbone, progress_cb=cb)
                return (
                    _weights_status_text(),
                    gr.update(visible=_any_weights_missing()),
                    gr.update(visible=_any_weights_missing()),
                    f"✅ Poids téléchargés : `models/rvm_{backbone}.pth`",
                )
            except Exception as e:
                return (
                    _weights_status_text(),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    f"❌ Erreur de téléchargement : {e}",
                )

        dl_btn.click(
            fn=on_download_weights,
            inputs=[dl_backbone],
            outputs=[weights_status_md, dl_btn, dl_backbone, dl_status],
        )

        # ── Launch inference ──
        def on_launch(
            video_path,
            backbone_label,
            resolution_label,
            manual_ratio,
            seq_chunk,
            alpha_gamma,
            format_label,
            selected_outputs,
            bg_name,
            output_dir_str,
            progress=gr.Progress(),
        ):
            global _cancel_event, _current_output_dir

            if not video_path or not Path(video_path).exists():
                yield (
                    "❌ Format non reconnu. Formats acceptés : MP4, MOV, MKV, AVI.",
                    "",
                    gr.update(visible=False),
                    [None]*4,
                    gr.update(), gr.update(),
                )
                return

            if not selected_outputs:
                yield (
                    "❌ Sélectionnez au moins une sortie (Composition, Alpha ou Foreground).",
                    "",
                    gr.update(visible=False),
                    [None]*4,
                    gr.update(), gr.update(),
                )
                return

            backbone = BACKBONE_MAP.get(backbone_label, "mobilenetv3")
            weights  = find_model_weights(backbone)
            if weights is None:
                yield (
                    "❌ Modèle introuvable. Cliquez sur 'Télécharger les poids' ci-dessus.",
                    "",
                    gr.update(visible=False),
                    [None]*4,
                    gr.update(), gr.update(),
                )
                return

            # Resolve downsample
            ds_val = RESOLUTION_MAP.get(resolution_label)
            if ds_val == "manual":
                ds_val = float(manual_ratio)

            # Output dir
            if output_dir_str and output_dir_str.strip():
                out_dir = Path(output_dir_str.strip())
            else:
                out_dir = Path.home() / "Desktop" / (Path(video_path).stem + "_RVM_output")
            _current_output_dir = out_dir

            # Outputs list
            outputs_list = [OUTPUT_NAMES[k] for k in selected_outputs if k in OUTPUT_NAMES]
            fmt = OUTPUT_FORMAT_MAP.get(format_label, "video")
            bg  = BG_MAP.get(bg_name, "black")
            device_str, _ = detect_device()

            # Reset cancel
            _cancel_event.clear()
            cancel = _cancel_event

            log_lines = []
            result_holder = {}
            error_holder  = {}

            def run():
                try:
                    from rvm_inference import RVMInference
                    inf = RVMInference(backbone=backbone, device=device_str)

                    def pb(frac, msg):
                        _progress_queue.put(("progress", frac, msg))

                    result = inf.process_video(
                        input_path   = Path(video_path),
                        output_dir   = out_dir,
                        downsample   = ds_val,
                        seq_chunk    = int(seq_chunk),
                        output_type  = fmt,
                        outputs      = outputs_list,
                        bg           = bg,
                        cancel_event = cancel,
                        progress_cb  = pb,
                        alpha_gamma  = float(alpha_gamma),
                    )
                    result_holder.update(result)
                except Exception as e:
                    error_holder["err"] = str(e)
                finally:
                    _progress_queue.put(("done", None, None))

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            # Poll the queue and yield updates
            while True:
                try:
                    item = _progress_queue.get(timeout=0.3)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    yield (
                        "\n".join(log_lines[-20:]) if log_lines else "Démarrage…",
                        "",
                        gr.update(visible=False),
                        [None]*4,
                        gr.update(), gr.update(),
                    )
                    continue

                kind, frac, msg = item
                if kind == "progress":
                    log_lines.append(msg)
                    progress(frac, desc=msg)
                    yield (
                        "\n".join(log_lines[-20:]),
                        f"**En cours…** {msg}",
                        gr.update(visible=False),
                        [None]*4,
                        gr.update(), gr.update(),
                    )
                elif kind == "done":
                    break

            thread.join(timeout=5)

            if "err" in error_holder:
                err = error_holder["err"]
                if "mémoire" in err.lower() or "memory" in err.lower():
                    msg = "⚠️ Mémoire insuffisante. Essayez : réduire le seq_chunk, passer au modèle Rapide, ou choisir la résolution Rapide — 0.25."
                else:
                    msg = f"❌ Erreur : {err}"
                yield (msg, msg, gr.update(visible=False), [None]*4, gr.update(), gr.update())
                return

            if cancel.is_set():
                msg = "🛑 Traitement annulé. Les frames déjà traitées sont disponibles dans le dossier de sortie."
                yield (msg, msg, gr.update(visible=False), [None]*4, gr.update(), gr.update())
                return

            # Build summary
            elapsed = result_holder.get("elapsed_s", 0)
            avg_fps = result_holder.get("avg_fps", 0)
            files   = result_holder.get("files", [])
            mins, secs = divmod(int(elapsed), 60)
            t_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            file_lines = "\n".join(f"  • `{p}` {sz}" for p, sz in files)
            summary = (
                f"✅ **Terminé en {t_str}** — {avg_fps:.1f} FPS moyen\n\n"
                f"**Fichiers générés :**\n{file_lines}"
            )

            # macOS notification
            notify_macos("RVM — Terminé", f"Traitement terminé en {t_str}")

            # Load result preview frames + source frames for side-by-side display
            stem = Path(video_path).stem
            res_frames  = _load_result_frames(out_dir, stem, bg_name)
            src_frames  = _source_frames(video_path)
            src_frame_0 = src_frames[0] if src_frames else None
            res_frame_0 = res_frames[0] if res_frames else None

            yield (
                "\n".join(log_lines[-20:]),
                summary,
                gr.update(visible=True),
                res_frames,
                src_frame_0,
                res_frame_0,
            )

        launch_btn.click(
            fn=on_launch,
            inputs=[
                video_input,
                backbone_radio,
                resolution_radio,
                manual_ds,
                seq_chunk_slider,
                alpha_sharpness_slider,
                output_format_radio,
                outputs_check,
                bg_radio,
                output_dir_box,
            ],
            outputs=[log_box, result_md, finder_row, result_frames_state, source_img, result_img],
        )

        # ── Cancel ──
        def on_cancel():
            _cancel_event.set()
            return "🛑 Annulation demandée…"

        cancel_btn.click(fn=on_cancel, outputs=[progress_md])

        # ── Open in Finder ──
        def on_open_finder():
            if _current_output_dir and _current_output_dir.exists():
                open_in_finder(_current_output_dir)

        finder_btn.click(fn=on_open_finder)

        # ── Preview navigation ──
        POSITIONS = ["10%", "25%", "50%", "75%"]

        def show_frame(idx, src_frames, res_frames):
            idx = int(idx) % 4
            src = src_frames[idx] if src_frames and idx < len(src_frames) else None
            res = res_frames[idx] if res_frames and idx < len(res_frames) else None
            label = f"**Frame {idx+1}/4** — {POSITIONS[idx]}"
            return idx, label, src, res

        def on_prev(idx, src_frames, res_frames):
            new_idx = (int(idx) - 1) % 4
            return show_frame(new_idx, src_frames, res_frames)

        def on_next(idx, src_frames, res_frames):
            new_idx = (int(idx) + 1) % 4
            return show_frame(new_idx, src_frames, res_frames)

        prev_btn.click(
            fn=on_prev,
            inputs=[preview_idx_state, src_frames_state, result_frames_state],
            outputs=[preview_idx_state, frame_label, source_img, result_img],
        )
        next_btn.click(
            fn=on_next,
            inputs=[preview_idx_state, src_frames_state, result_frames_state],
            outputs=[preview_idx_state, frame_label, source_img, result_img],
        )

        # Show first source frame immediately after upload
        def refresh_preview(src_frames):
            if src_frames and src_frames[0] is not None:
                return 0, "**Frame 1/4** — 10%", src_frames[0], None
            return 0, "**Frame 1/4** — 10%", None, None

        src_frames_state.change(
            fn=refresh_preview,
            inputs=[src_frames_state],
            outputs=[preview_idx_state, frame_label, source_img, result_img],
        )

        # When results arrive, refresh preview at current index
        def refresh_result_preview(idx, src_frames, res_frames):
            return show_frame(idx, src_frames, res_frames)

        result_frames_state.change(
            fn=refresh_result_preview,
            inputs=[preview_idx_state, src_frames_state, result_frames_state],
            outputs=[preview_idx_state, frame_label, source_img, result_img],
        )

    return demo


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _weights_status_text() -> str:
    lines = []
    for bb in ("mobilenetv3", "resnet50"):
        w = find_model_weights(bb)
        if w:
            icon = '<span class="status-ok">✓</span>'
        else:
            icon = '<span class="status-err">✕</span>'
        name = "MobileNetV3" if bb == "mobilenetv3" else "ResNet50"
        lines.append(f"{icon} {name}")
    return " · ".join(lines)


def _any_weights_missing() -> bool:
    return any(find_model_weights(bb) is None for bb in ("mobilenetv3", "resnet50"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_api=False,
    )
