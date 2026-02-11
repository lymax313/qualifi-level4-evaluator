from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Qualifi Level 4 Evaluator", version="4.0.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
