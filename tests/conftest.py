import sys
from pathlib import Path

# Ensure project root on sys.path for test imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
