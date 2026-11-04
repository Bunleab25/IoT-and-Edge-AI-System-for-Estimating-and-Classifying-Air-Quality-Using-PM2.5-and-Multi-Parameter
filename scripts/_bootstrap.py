"""Make the repository's src-layout package importable from script entry points."""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
