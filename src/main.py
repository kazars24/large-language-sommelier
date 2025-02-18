import uvicorn
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
