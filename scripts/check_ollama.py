"""Verify Ollama is running and required models are pulled."""
import sys
import requests

# Add project root to path so we can import from src/
sys.path.insert(0, ".")
from src.config import CFG

OLLAMA_URL = "http://localhost:11434"


def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_pulled_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags")
    return [m["name"].split(":")[0] for m in r.json().get("models", [])]


def main() -> int:
    if not check_ollama_running():
        print("❌ Ollama is not running. Start it (open the Ollama app or run `ollama serve`).")
        return 1
    print("✅ Ollama is running.")

    pulled = list_pulled_models()
    required = {
        CFG.models.generator.split(":")[0],
        CFG.models.grader.split(":")[0],
        CFG.models.embedder.split(":")[0],
    }
    missing = required - set(pulled)

    if missing:
        print(f"❌ Missing Ollama models: {missing}")
        print("   Pull them with:")
        for m in missing:
            print(f"     ollama pull {m}")
        return 1
    print(f"✅ All required Ollama models present: {required}")
    return 0


if __name__ == "__main__":
    sys.exit(main())