from pydantic import BaseModel, Field, ValidationInfo, field_validator

class RouteRequestSchema(BaseModel):
    """
    Strict payload validation to prevent injection vectors and out-of-bound graph traversal.
    """
    start_node_idx: int = Field(..., ge=0, description="Start node array index")
    end_node_idx: int = Field(..., ge=0, description="Destination node array index")
    total_nodes: int = Field(..., gt=0, description="Current graph node boundary limit")

    @field_validator('start_node_idx', 'end_node_idx')
    @classmethod
    def validate_graph_bounds(cls, value: int, info: ValidationInfo) -> int:
        total = info.data.get('total_nodes')
        if total is not None and value >= total:
            raise ValueError(
                f"Security Exception: Node index {value} exceeds memory boundary (max {total - 1}). "
                "Potential OOB/SSRF traversal attempt blocked."
            )
        return value
