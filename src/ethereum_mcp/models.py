"""Pydantic models for MCP tool input validation."""

from pydantic import BaseModel, Field, field_validator


class SearchInput(BaseModel):
    """Input validation for search operations."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    fork: str | None = Field(None, max_length=50, description="Fork filter")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")

    @field_validator("fork")
    @classmethod
    def validate_fork(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid_forks = [
            "phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"
        ]
        v_lower = v.lower()
        if v_lower not in valid_forks:
            raise ValueError(f"Invalid fork '{v}'. Must be one of: {valid_forks}")
        return v_lower


class ConstantLookupInput(BaseModel):
    """Input validation for constant lookup."""

    constant_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Constant name (UPPER_SNAKE_CASE)",
    )
    fork: str | None = Field(None, max_length=50, description="Fork filter")

    @field_validator("fork")
    @classmethod
    def validate_fork(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid_forks = [
            "phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"
        ]
        v_lower = v.lower()
        if v_lower not in valid_forks:
            raise ValueError(f"Invalid fork '{v}'. Must be one of: {valid_forks}")
        return v_lower


class FunctionAnalysisInput(BaseModel):
    """Input validation for function analysis."""

    function_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Function name (snake_case)",
    )
    fork: str | None = Field(None, max_length=50, description="Fork filter")

    @field_validator("fork")
    @classmethod
    def validate_fork(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid_forks = [
            "phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"
        ]
        v_lower = v.lower()
        if v_lower not in valid_forks:
            raise ValueError(f"Invalid fork '{v}'. Must be one of: {valid_forks}")
        return v_lower


class GuidanceInput(BaseModel):
    """Input validation for expert guidance lookup."""

    topic: str = Field(..., min_length=1, max_length=50, description="Guidance topic")


class ClientLookupInput(BaseModel):
    """Input validation for client lookup."""

    name: str = Field(..., min_length=1, max_length=50, description="Client name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Normalize to lowercase for matching
        return v.lower().strip()


class ClientListInput(BaseModel):
    """Input validation for client list."""

    layer: str | None = Field(
        None,
        description="Layer filter: 'execution' or 'consensus'",
    )

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v_lower = v.lower().strip()
        if v_lower not in ("execution", "consensus"):
            raise ValueError("Layer must be 'execution' or 'consensus'")
        return v_lower


class EipSearchInput(BaseModel):
    """Input validation for EIP search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    eip_number: str | None = Field(
        None,
        max_length=10,
        pattern=r"^\d+$",
        description="EIP number filter",
    )
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")
