"""Strategy Manager — FastAPI server entry point."""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from manager.api import router

app = FastAPI(title="Strategy Manager", version="1.0")
app.include_router(router, prefix="/api")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

@app.get("/")
def index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/app.js")
def app_js():
    return FileResponse(os.path.join(TEMPLATES_DIR, "app.js"))

if __name__ == "__main__":
    print("Starting Strategy Manager on http://localhost:5000")
    print("Default password: password123 (edit manager/.secret to change)")
    import os
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), 'key.pem')
    ssl_cert = (cert_file, key_file) if os.path.exists(cert_file) and os.path.exists(key_file) else None
    if ssl_cert:
        print("HTTPS enabled with self-signed cert")
    uvicorn.run(app, host="0.0.0.0", port=5000, ssl_certfile=cert_file if ssl_cert else None,
                ssl_keyfile=key_file if ssl_cert else None)
