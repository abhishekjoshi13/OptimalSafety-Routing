
from pydantic import BaseModel, Field, field_validator


class LocationQuerySchema(BaseModel):
    """
    Validates and normalizes user-provided city and route locations.
    """

    city_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Target city or area used for graph generation"
    )

    start_address: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Starting location"
    )

    end_address: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Destination location"
    )

    @field_validator(
        "city_name",
        "start_address",
        "end_address",
        mode="before"
    )
    @classmethod
    def normalize_location_text(cls, value):
        if not isinstance(value, str):
            raise ValueError("Location input must be a string")

        value = value.strip()

        if not value:
            raise ValueError("Location input cannot be empty")

        return value
