import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

FALKORDB_HOST: str = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT: int = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH: str = os.getenv("FALKORDB_GRAPH", "badi_trishuli_graph")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
if not OPENROUTER_API_KEY and GEMINI_API_KEY.startswith("sk-or-"):
    OPENROUTER_API_KEY = GEMINI_API_KEY
    GEMINI_API_KEY = ""

OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free").strip()
