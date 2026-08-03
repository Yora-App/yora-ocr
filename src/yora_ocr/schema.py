from pydantic import BaseModel
from datetime import datetime


class ShopItem(BaseModel):
    price: int
    amount: float = 1.0
    unit: str = "piece"
    price_per_unit: int


class Purchase(BaseModel):
    shop: str | None = None
    date: datetime | None = None
    items: dict[str, ShopItem]
    total: int
