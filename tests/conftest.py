import sys
from pathlib import Path

# drone_nav without installing it (its top-level package doesn't import Isaac Lab).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "drone_nav"))
