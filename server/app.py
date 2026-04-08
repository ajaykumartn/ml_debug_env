# Entry point for openenv validate — imports the main app
from app import app

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
