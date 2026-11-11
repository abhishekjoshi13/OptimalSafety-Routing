from pydantic import BaseModel, Field

class LocationQuerySchema(BaseModel):
    """
    Validates user text inputs for city and routing locations to prevent Geocoding API crashes.
    """
    city_name: str = Field(..., min_length=2, description="Target city for bounding box")
    start_address: str = Field(..., min_length=2, description="Origin point")
    end_address: str = Field(..., min_length=2, description="Destination point")
