
import os
import sys

# Prevent sentence-transformers / HuggingFace from making network calls.
# The model is already cached locally; this avoids proxy/firewall errors on startup.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.web.app import app

if __name__ == "__main__":
    print("Starting FineTuner Web App...")
    # Enable reloader and debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
