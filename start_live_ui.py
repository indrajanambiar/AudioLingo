#!/usr/bin/env python3
"""Start the AudioLingo Live UI server."""
import uvicorn
from ui.live_ui import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
