from pydantic import BaseModel, ConfigDict, create_model

from src.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


class _FlowBase(BaseModel):
    """Reject missing and unexpected fields in raw flow records."""

    model_config = ConfigDict(extra="forbid")


Flow = create_model(
    "Flow",
    __base__=_FlowBase,
    **{
        name: (str if name in CATEGORICAL_COLUMNS else float, ...)
        for name in FEATURE_COLUMNS
    },
)


class PredictRequest(BaseModel):
    flows: list[Flow]


class Prediction(BaseModel):
    label: str
    is_attack: bool
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float

