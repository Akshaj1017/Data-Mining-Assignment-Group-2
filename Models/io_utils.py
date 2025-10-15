from pathlib import Path

def ensure_output_dir(out_dir="output"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
