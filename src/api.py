from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup — load model into memory here
    yield
    # runs at shutdown — cleanup if needed
    
    
    
class TransactionRequest(BaseModel):
    TransactionAmt: float
    card1: int
    addr1: float
    TransactionDT: int
    ProductCD: str