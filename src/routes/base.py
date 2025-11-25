from fastapi import FastAPI ,APIRouter
from dotenv import load_dotenv
import os

load_dotenv()


base_router = APIRouter(
    prefix="/api/v1",
    tags=["base check"],
)
 

@base_router.get("/")
async def read_root():
    return {"app_name": os.getenv("APP_NAME"),
            "app_version": os.getenv("APP_VERSION")}

 