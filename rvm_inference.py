"""
RVM inference engine — isolated from Gradio, testable in CLI.

Usage:
    python rvm_inference.py --input video.mp4 --output ./out \
        [--backbone mobilenetv3] [--downsample auto] \
        [--seq-chunk 12] [--outputs composite,alpha,foreground] \
        [--format video|png|both] [--bg black|white|green|checker]
"""

import os
# hardsigmoid is not natively supported on MPS in torch 2.x — enable CPU fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import collections
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader

from rvm_model import MattingNetwork
from utils import find_model_weights


# ---------------------------------------------------------------------------
# MPS-friendly Hardswish / Hardsigmoid
# ---------------------------------------------------------------------------
#
# nn.Hardsigmoid → aten::hardsigmoid is not implemented on MPS in torch 2.0.x,
# which forces PYTORCH_ENABLE_MPS_FALLBACK to ferry every call through the CPU.
# MobileNetV3-Large contains 9 Hardswish blocks (each one is a Hardsigmoid)
# plus Squeeze-Excitation modules that also use Hardsigmoid — so a single
# forward pass triggers dozens of MPS→CPU→MPS round-trips. The pipeline
# saturates after a couple of batches and each subsequent forward takes 20–30 s
# instead of <1 s.
#
# These drop-in replacements use only relu6/mul, which are native on MPS.
# Same maths, zero fallback. Hardswish(x) = x * relu6(x+3) / 6.
class _MPSHardsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class _MPSHardswish(nn.Module):
    def forward(self, x):
        return x * (F.relu6(x + 3.0) / 6.0)


def _patch_hardops(module: nn.Module) -> None:
    """Recursively replace nn.Hardswish / nn.Hardsigmoid in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, _MPSHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, _MPSHardsigmoid())
        else:
            _patch_hardops(child)


def _mps_empty_cache() -> None:
    """Free MPS allocator cache — API name moved between torch 2.0 and 2.1+."""
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif hasattr(torch._C, "_mps_emptyCache"):
        torch._C._mps_emptyCache()


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

BG_COLORS = {
    "black":   (0.0,   0.0,   0.0),
    "white":   (1.0,   1.0,   1.0),
    "green":   (0.0,   0.694, 0.251),  # #00b140
}

def _make_checker(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a [1, 1, 3, H, W] checker-board tensor."""
    cell = 32
    img = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(0, h, cell):
        for col in range(0, w, cell):
            light = ((row // cell) + (col // cell)) % 2 == 0
            color = 0.75 if light else 0.5
            img[row:row+cell, col:col+cell] = color
    t = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return t.unsqueeze(0).unsqueeze(0).to(device, dtype)  # [1,1,3,H,W]


def _solid_bg(color_tuple, h: int, w: int, device, dtype) -> torch.Tensor:
    r, g, b = color_tuple
    return torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 3, 1, 1)


# ---------------------------------------------------------------------------
# cv2-based Video Dataset (sequential streaming)
# ---------------------------------------------------------------------------

class VideoDataset(IterableDataset):
    """
    Sequential video reader — one persistent VideoCapture, no seek per frame.

    Map-style + CAP_PROP_POS_FRAMES forces opencv to decode from the previous
    keyframe (or the start of the file) on every read, which is O(N) per frame
    and O(N²) overall. Streaming keeps a single capture open and reads in order.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir : {path}")
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        cap = cv2.VideoCapture(str(self.path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # BGR uint8 HWC → RGB float32 CHW in [0,1], no PIL round-trip
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# cv2-based Video writer
# ---------------------------------------------------------------------------

class _VideoWriter:
    """RGB video writer. Expects CPU uint8 tensors of shape [T, C, H, W]."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vw = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def write_cpu_batch(self, cpu_uint8: torch.Tensor):
        arr = cpu_uint8.permute(0, 2, 3, 1).numpy()
        for t in range(arr.shape[0]):
            self.vw.write(cv2.cvtColor(arr[t], cv2.COLOR_RGB2BGR))

    def close(self):
        self.vw.release()


class _PngWriter:
    """RGB PNG sequence writer. Expects CPU uint8 tensors of shape [T, C, H, W]."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self.counter = 0

    def write_cpu_batch(self, cpu_uint8: torch.Tensor):
        arr = cpu_uint8.permute(0, 2, 3, 1).numpy()
        for t in range(arr.shape[0]):
            Image.fromarray(arr[t]).save(self.dir / f"frame_{self.counter:05d}.png")
            self.counter += 1

    def close(self):
        pass


class _AlphaPngWriter:
    """Grayscale alpha PNG writer. Expects CPU uint8 tensors of shape [T, 1, H, W]."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self.counter = 0

    def write_cpu_batch(self, cpu_uint8: torch.Tensor):
        arr = cpu_uint8.squeeze(1).numpy()
        for t in range(arr.shape[0]):
            Image.fromarray(arr[t], mode='L').save(self.dir / f"alpha_{self.counter:05d}.png")
            self.counter += 1

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Auto downsample
# ---------------------------------------------------------------------------

def auto_downsample_ratio(h: int, w: int) -> float:
    return min(512 / max(h, w), 1.0)


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class RVMInference:
    def __init__(self, backbone: str, device: str):
        weights = find_model_weights(backbone)
        if weights is None:
            raise FileNotFoundError(
                f"❌ Modèle introuvable. Cliquez sur 'Télécharger les poids' ci-dessus."
            )
        self.device = torch.device(device)
        self.model = MattingNetwork(backbone).eval().to(self.device)
        state = torch.load(str(weights), map_location=self.device)
        self.model.load_state_dict(state)
        # Replace Hardswish/Hardsigmoid with MPS-native equivalents — see
        # _MPSHardswish docstring. Safe on CPU/CUDA too: the maths is identical.
        if self.device.type == "mps":
            _patch_hardops(self.model)

    def process_video(
        self,
        input_path:   Path,
        output_dir:   Path,
        downsample:   Optional[float],     # None = auto
        seq_chunk:    int,
        output_type:  str,                 # 'video' | 'png' | 'both'
        outputs:      List[str],           # ['composite','alpha','foreground']
        bg:           str,                 # 'black'|'white'|'green'|'checker'
        cancel_event: threading.Event,
        progress_cb:  Optional[Callable],  # (fraction, message) -> None
    ) -> dict:
        """
        Run RVM on input_path, write results to output_dir.
        Returns dict with keys: files, elapsed_s, avg_fps.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset   = VideoDataset(input_path)
        fps       = dataset.frame_rate
        n_frames  = len(dataset)
        W, H      = dataset.width, dataset.height
        dtype     = torch.float32

        loader = DataLoader(
            dataset,
            batch_size=seq_chunk,
            num_workers=0,       # 0 = main thread; avoids multiprocessing on MPS
            pin_memory=False,
        )

        # Resolve downsample
        if downsample is None:
            ds_ratio = auto_downsample_ratio(H, W)
        else:
            ds_ratio = downsample

        # Background tensor (lazy — resolved on first frame so we know H/W)
        bg_tensor = None

        # Open writers
        def _stem(suffix): return output_dir / f"{Path(input_path).stem}_{suffix}"

        writers_composite  = []
        writers_alpha      = []
        writers_foreground = []

        if 'composite' in outputs:
            if output_type in ('video', 'both'):
                writers_composite.append(_VideoWriter(_stem('composite.mp4'), fps, W, H))
            if output_type in ('png', 'both'):
                writers_composite.append(_PngWriter(_stem('composite_png')))

        if 'alpha' in outputs:
            if output_type in ('video', 'both'):
                writers_alpha.append(_VideoWriter(_stem('alpha.mp4'), fps, W, H))
            if output_type in ('png', 'both'):
                writers_alpha.append(_AlphaPngWriter(_stem('alpha_png')))

        if 'foreground' in outputs:
            if output_type in ('video', 'both'):
                writers_foreground.append(_VideoWriter(_stem('foreground.mp4'), fps, W, H))
            if output_type in ('png', 'both'):
                writers_foreground.append(_PngWriter(_stem('foreground_png')))

        all_writers = writers_composite + writers_alpha + writers_foreground
        generated_files = []

        start_time = time.time()
        frames_done = 0
        # Sliding window of (frames_done, timestamp) samples — gives an FPS
        # that reflects current speed instead of being dragged down by the
        # cold-start warmup batch.
        recent = collections.deque(maxlen=6)
        recent.append((0, start_time))

        # ---- Inference loop ----
        rec = [None] * 4  # recurrent state — initialised ONCE before the loop

        try:
            with torch.inference_mode():
                for src_batch in loader:
                    if cancel_event.is_set():
                        break

                    # src_batch: [T, C, H, W] → unsqueeze B dim → [1, T, C, H, W]
                    src = src_batch.to(self.device, dtype, non_blocking=False).unsqueeze(0)
                    T = src.shape[1]

                    fgr, pha, *rec = self.model(src, *rec, ds_ratio)
                    # fgr, pha: [1, T, C, H, W]

                    # Detach rec to cut the autograd graph — prevents MPS memory accumulation
                    rec = [t.detach() if isinstance(t, torch.Tensor) else t for t in rec]

                    # Composite background.
                    com = None
                    if writers_composite:
                        if bg == 'checker':
                            if bg_tensor is None:
                                bg_tensor = _make_checker(H, W, self.device, dtype)
                        else:
                            if bg_tensor is None:
                                bg_tensor = _solid_bg(BG_COLORS.get(bg, (0,0,0)), H, W, self.device, dtype)
                        com = fgr * pha + bg_tensor * (1 - pha)  # [1,T,3,H,W]

                    # Move every output the writers need to CPU as uint8 *before*
                    # iterating again. Holding full-res MPS tensors across
                    # iterations pushes an 8 GB unified-memory Mac into swap, and
                    # subsequent batches go from ~1 s to 10–25 s.
                    com_cpu = com[0].mul(255).clamp(0, 255).byte().cpu() if com is not None else None
                    pha_cpu_gray = pha[0].mul(255).clamp(0, 255).byte().cpu() if writers_alpha else None
                    fgr_cpu = fgr[0].mul(255).clamp(0, 255).byte().cpu() if writers_foreground else None

                    # Drop the MPS-side tensors immediately, then flush the
                    # allocator so the next batch starts from a clean slate.
                    del fgr, pha, src
                    if com is not None:
                        del com
                    if self.device.type == "mps":
                        _mps_empty_cache()

                    # Write outputs (CPU-only from here on, fully decoupled from MPS).
                    for w in writers_composite:
                        w.write_cpu_batch(com_cpu)
                    for w in writers_alpha:
                        if isinstance(w, _AlphaPngWriter):
                            w.write_cpu_batch(pha_cpu_gray)
                        else:
                            # Video alpha = replicate gray to 3 channels (on CPU)
                            w.write_cpu_batch(pha_cpu_gray.repeat(1, 3, 1, 1))
                    for w in writers_foreground:
                        w.write_cpu_batch(fgr_cpu)

                    frames_done += T

                    now = time.time()
                    recent.append((frames_done, now))
                    window_frames = recent[-1][0] - recent[0][0]
                    window_time   = recent[-1][1] - recent[0][1]
                    current_fps   = window_frames / window_time if window_time > 0 else 0
                    remaining = (n_frames - frames_done) / current_fps if current_fps > 0 else 0
                    mins, secs = divmod(int(remaining), 60)
                    eta = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

                    if progress_cb:
                        progress_cb(
                            frames_done / n_frames,
                            f"Frame {frames_done}/{n_frames} — {current_fps:.1f} FPS — Temps restant : {eta}"
                        )

        finally:
            for w in all_writers:
                w.close()

        elapsed_total = time.time() - start_time
        avg_fps_final = frames_done / elapsed_total if elapsed_total > 0 else 0

        # Collect generated file sizes
        for w in all_writers:
            if hasattr(w, 'vw'):  # VideoWriter
                p = w.vw  # already released
            elif isinstance(w, (_PngWriter, _AlphaPngWriter)):
                if w.dir.exists():
                    generated_files.append((str(w.dir), ""))
                continue
        # Scan output_dir for actual files
        generated_files = []
        for p in sorted(output_dir.iterdir()):
            if p.is_file() and p.suffix in ('.mp4', '.mov'):
                sz = p.stat().st_size / (1024 * 1024)
                generated_files.append((str(p), f"{sz:.1f} Mo"))
            elif p.is_dir():
                count = sum(1 for _ in p.glob('*.png'))
                generated_files.append((str(p), f"{count} images PNG"))

        return {
            "files":     generated_files,
            "elapsed_s": elapsed_total,
            "avg_fps":   avg_fps_final,
            "cancelled": cancel_event.is_set(),
        }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    default="./rvm_out")
    parser.add_argument("--backbone",  default="mobilenetv3", choices=["mobilenetv3", "resnet50"])
    parser.add_argument("--downsample", default=None, type=float)
    parser.add_argument("--seq-chunk", default=12, type=int)
    parser.add_argument("--outputs",   default="composite,alpha,foreground")
    parser.add_argument("--format",    default="video", choices=["video", "png", "both"])
    parser.add_argument("--bg",        default="black", choices=["black", "white", "green", "checker"])
    parser.add_argument("--device",    default=None)
    args = parser.parse_args()

    from utils import detect_device
    device_str, device_msg = detect_device()
    if args.device:
        device_str = args.device
    print(f"Device : {device_msg}")

    inf = RVMInference(backbone=args.backbone, device=device_str)

    cancel = threading.Event()
    result = inf.process_video(
        input_path   = Path(args.input),
        output_dir   = Path(args.output),
        downsample   = args.downsample,
        seq_chunk    = args.seq_chunk,
        output_type  = args.format,
        outputs      = args.outputs.split(","),
        bg           = args.bg,
        cancel_event = cancel,
        progress_cb  = lambda f, msg: print(f"  {msg}"),
    )

    print(f"\n✅ Terminé en {result['elapsed_s']:.1f}s — {result['avg_fps']:.1f} FPS moyen")
    for path, size in result["files"]:
        print(f"  {path}  {size}")
