from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

from app.routers import predictions
from app.utils.logger import setup_logger

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja loggera
logger = setup_logger(__name__)

# Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Forex AI Signal API",
    description="API do generowania sygnałów tradingowych z wykorzystaniem modeli AI",
    version="1.0.0"
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji ogranicz do konkretnych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dodanie routerów
app.include_router(predictions.router)

@app.get("/")
async def root():
    return {"message": "Forex AI Signal API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)