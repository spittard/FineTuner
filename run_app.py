
import os
import sys

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.web.app import app

if __name__ == "__main__":
    print("Starting FineTuner Web App...")
    # Enable reloader and debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
