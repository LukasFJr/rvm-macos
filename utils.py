import subprocess
import sys
import os
from pathlib import Path
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> "tuple[str, str]":
    """Return (device_str, human_readable_message)."""
    try:
        import torch
    except ImportError:
        return "cpu", "⚠️ PyTorch non installé — CPU uniquement"

    if torch.backends.mps.is_available():
        return "mps", "⚡ Apple Silicon MPS détecté"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", f"⚡ CUDA détecté : {name}"
    return "cpu", "🐢 CPU uniquement (traitement plus lent)"


def get_env_info() -> str:
    """Return a Markdown string with Python, PyTorch version, and device."""
    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "non installé"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    device, device_msg = detect_device()

    return (
        f"**Python** {py_ver} · **PyTorch** {torch_ver}\n\n"
        f"{device_msg}"
    )


# ---------------------------------------------------------------------------
# Video info
# ---------------------------------------------------------------------------

def get_video_info(path: Path) -> dict:
    """
    Return dict with keys: width, height, fps, frame_count, duration_s,
    codec, file_size_mb, thumbnail (np.ndarray RGB or None).
    """
    try:
        import cv2
    except ImportError:
        return {"error": "opencv-python non installé"}

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": f"Impossible d'ouvrir : {path.name}"}

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps > 0 else 0.0

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

    # First frame thumbnail
    ok, frame = cap.read()
    thumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
    cap.release()

    file_size_mb = path.stat().st_size / (1024 * 1024)

    mins, secs = divmod(int(duration), 60)
    duration_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    return {
        "width":         width,
        "height":        height,
        "fps":           fps,
        "frame_count":   n_frames,
        "duration_s":    duration,
        "duration_str":  duration_str,
        "codec":         codec,
        "file_size_mb":  round(file_size_mb, 1),
        "thumbnail":     thumbnail,
        "name":          path.name,
    }


def format_video_info_md(info: dict) -> str:
    """Convert get_video_info() result to a Markdown summary string."""
    if "error" in info:
        return f"❌ {info['error']}"
    return (
        f"**{info['name']}** · "
        f"{info['width']}×{info['height']} · "
        f"{info['fps']:.2f} FPS · "
        f"{info['frame_count']} frames · "
        f"{info['duration_str']} · "
        f"Codec: `{info['codec']}` · "
        f"{info['file_size_mb']} Mo"
    )


# ---------------------------------------------------------------------------
# macOS integrations
# ---------------------------------------------------------------------------

def open_in_finder(path: Path) -> None:
    subprocess.run(["open", str(path)], check=False)


def notify_macos(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False)


# ---------------------------------------------------------------------------
# Model weight management
# ---------------------------------------------------------------------------

WEIGHT_URLS = {
    "mobilenetv3": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
    "resnet50":    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
}

def find_model_weights(backbone: str) -> Optional[Path]:
    """Search for rvm_{backbone}.pth in ./models/ then current dir."""
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "models" / f"rvm_{backbone}.pth",
        script_dir / f"rvm_{backbone}.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def download_weights(backbone: str, progress_cb=None) -> Path:
    """
    Download model weights from GitHub Releases via urllib.
    progress_cb(fraction, message) called periodically if provided.
    Returns the saved path.
    """
    import urllib.request

    url = WEIGHT_URLS.get(backbone)
    if url is None:
        raise ValueError(f"Backbone inconnu : {backbone}")

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / f"rvm_{backbone}.pth"

    if progress_cb:
        progress_cb(0.0, f"Téléchargement de rvm_{backbone}.pth…")

    def _report(block_count, block_size, total_size):
        if total_size > 0 and progress_cb:
            done = min(block_count * block_size, total_size)
            progress_cb(done / total_size, f"Téléchargement… {done // (1024*1024)} Mo / {total_size // (1024*1024)} Mo")

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)

    if progress_cb:
        progress_cb(1.0, "Téléchargement terminé ✅")

    return dest


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Environnement ===")
    print(get_env_info())

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        print(f"\n=== Info vidéo : {video_path} ===")
        info = get_video_info(video_path)
        print(format_video_info_md(info))
    else:
        print("\nUsage: python utils.py [chemin_video]")

    print("\n=== Poids du modèle ===")
    for bb in ("mobilenetv3", "resnet50"):
        w = find_model_weights(bb)
        print(f"  {bb}: {'✅ ' + str(w) if w else '❌ absent'}")
