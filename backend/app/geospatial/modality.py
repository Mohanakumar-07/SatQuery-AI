"""Optical vs SAR modality detection.

The router refuses to guess between a bi-temporal pair and an optical-SAR pair
(plan section 9), so modality evidence needs one shared implementation with an
explicit certainty value. Band names and sensor identity are strong evidence; band
count alone is deliberately **not** — a 1-band raster is equally likely to be SAR
backscatter or an optical panchromatic export, and treating it as either would
silently mis-route the analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.common import Modality

_SAR_BANDS = {"vv", "vh", "hh", "hv", "vv_db", "vh_db", "sigma0_hh", "sigma0_vv", "gamma0_vv", "gamma0_vh"}
_OPTICAL_BANDS = {
    "coastal",
    "blue",
    "green",
    "red",
    "rededge",
    "red_edge",
    "nir",
    "narrow_nir",
    "swir16",
    "swir22",
    "pan",
    "panchromatic",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "b7",
    "b8",
    "b8a",
    "b11",
    "b12",
    "band1",
    "band2",
    "band3",
    "band4",
}

_SAR_SENSORS = (
    "sentinel-1",
    "sentinel1",
    "s1_l1c",
    "s1_l2",
    "alos",
    "palsar",
    "terrasar",
    "cosmo",
    "radarsat",
    "nistar",
    "uavsar",
    "sir-c",
)
_OPTICAL_SENSORS = (
    "sentinel-2",
    "sentinel2",
    "landsat",
    "spot",
    "pleiades",
    "worldview",
    "icubes",
    "cartosat",
    "resourcesat",
    "resource_sat",
    "resourcessat",
    "modis",
    "viirs",
    "planet",
    "digitalglobe",
    "gaofen",
    "zhuhai",
    "kompsat",
)
_NON_DATE = re.compile(r"[^a-z0-9]+")


@dataclass
class ModalityGuess:
    modality: Modality
    #: Human-readable reasons, surfaced in validation warnings and the trace.
    evidence: list[str] = field(default_factory=list)
    certainty: float = 0.0

    @property
    def decided(self) -> bool:
        return self.modality is not Modality.OTHER and self.certainty >= 0.5


def _normalise(value: str | None) -> str:
    if not value:
        return ""
    return _NON_DATE.sub("", value.lower())


def _band_tokens(band_names) -> list[str]:
    tokens: list[str] = []
    for name in band_names or ():
        text = str(name or "").strip().lower()
        if not text:
            continue
        tokens.append(text)
        tokens.extend(part for part in re.split(r"[^a-z0-9]+", text) if part)
    return tokens


def infer_modality(
    *,
    sensor: str | None = None,
    band_names=None,
    declared: str | None = None,
    media_kind: str | None = None,
    filename: str | None = None,
) -> ModalityGuess:
    """Classify one raster as optical, SAR, or undecided."""
    evidence: list[str] = []

    if declared and declared not in {"unknown", "other", ""}:
        try:
            modality = Modality(declared.lower())
        except ValueError:
            modality = None
        if modality and modality is not Modality.OTHER:
            return ModalityGuess(modality, [f"client declared modality '{modality.value}'"], 0.9)
        if declared:
            evidence.append(f"ignored unusable declared modality '{declared}'")

    tokens = set(_band_tokens(band_names))
    if tokens & _SAR_BANDS:
        matched = sorted(tokens & _SAR_BANDS)
        return ModalityGuess(Modality.SAR, [f"polarisation band names {matched}"], 0.9)

    haystack = " ".join(filter(None, [_normalise(sensor), _normalise(filename)]))
    for marker in _SAR_SENSORS:
        if _normalise(marker) and _normalise(marker) in haystack:
            return ModalityGuess(Modality.SAR, [f"sensor '{marker}' is a radar platform"], 0.85)
    for marker in _OPTICAL_SENSORS:
        if _normalise(marker) and _normalise(marker) in haystack:
            return ModalityGuess(Modality.OPTICAL, [f"sensor '{marker}' is an optical platform"], 0.85)

    band_list = [str(name) for name in (band_names or ()) if name]
    if band_list and not (set(_band_tokens(band_list)) & _SAR_BANDS):
        if len(band_list) >= 3:
            return ModalityGuess(
                Modality.OPTICAL,
                [f"{len(band_list)} named non-polarisation bands", "no SAR polarisation present"],
                0.6,
            )
        evidence.append(f"{len(band_list)} named band(s) without polarisation markers")

    if tokens & _OPTICAL_BANDS and not (tokens & _SAR_BANDS):
        return ModalityGuess(Modality.OPTICAL, [f"optical band names {sorted(tokens & _OPTICAL_BANDS)}"], 0.6)

    return ModalityGuess(Modality.OTHER, evidence or ["no sensor, polarisation or band evidence"], 0.0)
