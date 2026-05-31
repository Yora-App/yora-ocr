from pydantic import BaseModel
from datetime import datetime

class ShopItem(BaseModel):
    price: float
    amount: float = 1.0
    unit: str = "piece" 
    price_per_unit: float 


class Purchase(BaseModel):
    shop: str | None = None
    date: datetime | None = None
    items: dict[str, ShopItem]
    total: float | None = None

