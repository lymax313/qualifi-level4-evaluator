from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Qualifi Level 4 Evaluator", version="4.0.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Qualifi Level 4 Evaluator</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
