"""Pydantic schemas for API I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-serialisable values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class Modality(str, Enum):
    image = "image"
    video = "video"
    audio = "audio"


class Verdict(str, Enum):
    likely_real = "likely_real"
    inconclusive = "inconclusive"
    likely_ai = "likely_ai"


class SignalSeverity(str, Enum):
    info = "info"
    warn = "warn"
    flag = "flag"
    pass_ = "pass"


class SignalResult(BaseModel):
    """One named signal output from any layer."""

    id: str
    layer: int
    name: str
    description: str = ""
    severity: SignalSeverity = SignalSeverity.info

    # Raw probability that the *signal* believes the input is AI-generated.
    # 0.0 = strong real, 1.0 = strong AI, 0.5 = no opinion.
    p_ai: float = 0.5
    confidence: float = 0.5  # how strongly to weight this signal (0..1)

    domain: Literal[
        "provenance", "quantum", "thermodynamic", "biological", "semantic", "ml"
    ] = "semantic"

    # Free-form structured details for the UI.
    evidence: Dict[str, Any] = Field(default_factory=dict)
    plain_language: str = ""

    error: Optional[str] = None

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: Any) -> Any:
        return _to_jsonable(v) if v is not None else {}

    @field_validator("p_ai", "confidence", mode="before")
    @classmethod
    def _coerce_floats(cls, v: Any) -> Any:
        if isinstance(v, np.generic):
            return float(v.item())
        return v


class DomainBreakdown(BaseModel):
    provenance: float = 0.5
    quantum: float = 0.5
    thermodynamic: float = 0.5
    biological: float = 0.5
    semantic: float = 0.5
    ml: float = 0.5


class HeatmapAsset(BaseModel):
    kind: Literal["gradcam", "noise", "fft", "block"] = "gradcam"
    mime: str = "image/png"
    data_base64: str
    description: str = ""


class TrailItem(BaseModel):
    """One structured entry in the forensic provenance trail."""

    layer: int
    id: str
    name: str
    severity: SignalSeverity
    p_ai: float
    confidence: float
    plain_language: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: Any) -> Any:
        return _to_jsonable(v) if v is not None else {}

    @field_validator("p_ai", "confidence", mode="before")
    @classmethod
    def _coerce_floats(cls, v: Any) -> Any:
        if isinstance(v, np.generic):
            return float(v.item())
        return v


class AnalyzeResponse(BaseModel):
    request_id: str
    modality: Modality
    filename: str
    bytes: int

    score: float = Field(..., ge=0.0, le=100.0, description="Authenticity 0..100")
    score_uncertainty: float = Field(..., ge=0.0, le=50.0)
    verdict: Verdict
    verdict_label: str

    p_ai: float = Field(..., ge=0.0, le=1.0)
    domain_real_confidence: DomainBreakdown

    forensic_trail: List[str]
    provenance_trail: List["TrailItem"] = Field(default_factory=list)
    checklist: List[str]
    signals: List[SignalResult]
    heatmaps: List[HeatmapAsset] = Field(default_factory=list)

    processing_ms: int
    model_versions: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    ml_enabled: bool
    device: str
