from fastapi import APIRouter, HTTPException, Depends, Header
import pandas as pd
from typing import Optional

from app.models.schemas import PredictionRequest, PredictionResponse
from app.models.predictor import ForexPredictor
from app.utils.logger import setup_logger

# Konfiguracja loggera
logger = setup_logger(__name__)

# Inicjalizacja routera
router = APIRouter(
    prefix="/api",
    tags=["predictions"],
)

# Inicjalizacja predyktora
predictor = ForexPredictor()

# Funkcja do weryfikacji klucza API
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        return None  # Na razie pomijamy weryfikację klucza API
    return authorization

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Otrzymano żądanie predykcji dla symbolu: {request.symbol}")
        
        # Konwersja danych do DataFrame
        df = pd.DataFrame([data.dict() for data in request.data])
        
        # Wywołanie predyktora
        prediction = predictor.predict(
            symbol=request.symbol,
            data=df,
            parameters=request.parameters
        )
        
        logger.info(f"Wygenerowano predykcję dla {request.symbol}: {prediction['type']}")
        
        # Zwróć odpowiedź
        return PredictionResponse(
            symbol=request.symbol,
            type=prediction["type"],
            entryPrice=prediction["entry_price"],
            tp1=prediction["tp1"],
            tp2=prediction["tp2"],
            sl=prediction["sl"],
            confidence=prediction["confidence"]
        )
    except Exception as e:
        logger.error(f"Błąd podczas generowania predykcji: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))