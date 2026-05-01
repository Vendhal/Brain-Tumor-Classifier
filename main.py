from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mcp_server import mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health check route (NOT a mount!)
@app.get("/")
async def root():
    return JSONResponse({
        "status": "healthy",
        "service": "Brain Tumor Classifier MCP Server",
        "mcp_endpoint": "/mcp"
    })

# Mount MCP server at /mcp
app.mount("/mcp", mcp.streamable_http_app())