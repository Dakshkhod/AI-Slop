"""Print each individual model's prediction for the user's AI image and a real photo."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.layer4_ml import _load_image_pipes, _load_clip, _interpret_predictions

AI_IMAGE = Path(
    r"C:/Users/DK/.cursor/projects/d-Slop-Detector/assets/"
    r"c__Users_DK_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"89f2837b043bf2aad9629a2fad021998_images_image-49bca521-7683-43db-bea7-3d81417b832f.png"
)


def _print_each(name: str, pil: Image.Image) -> None:
    print(f"\n{'=' * 60}\n=== {name}\n{'=' * 60}")
    for mid, pipe in _load_image_pipes():
        try:
            preds = pipe(pil)
            p_ai, table = _interpret_predictions(preds)
            print(f"  {mid:50s}  p(AI)={p_ai:.3f}")
            for label, score in sorted(table.items(), key=lambda kv: -kv[1])[:4]:
                print(f"    {label:25s}  {score:.3f}")
        except Exception as e:
            print(f"  {mid:50s}  FAILED: {e}")
    state = _load_clip()
    if state is not None:
        import torch  # type: ignore

        with torch.no_grad():
            inputs = state["processor"](images=pil, return_tensors="pt").to(state["device"])
            vision_out = state["model"].vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            feat = state["model"].visual_projection(pooled)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim_real = float((feat @ state["real_proto"].T).squeeze().item())
            sim_ai = float((feat @ state["ai_proto"].T).squeeze().item())
        margin = sim_ai - sim_real
        print(f"  CLIP                                                "
              f" sim_real={sim_real:.4f} sim_ai={sim_ai:.4f} margin={margin:+.4f}")


def main() -> int:
    if AI_IMAGE.exists():
        _print_each("user-ai-portrait.png (AI)", Image.open(AI_IMAGE).convert("RGB"))

    try:
        r = requests.get(
            "https://images.unsplash.com/photo-1494790108377-be9c29b29330"
            "?w=800&q=80&auto=format",
            timeout=8,
        )
        if r.status_code == 200:
            _print_each("real-unsplash.jpg (REAL)",
                        Image.open(io.BytesIO(r.content)).convert("RGB"))
    except Exception as e:
        print(f"unsplash fetch failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
