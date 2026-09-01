"""Evidence engine boundary (plan section 12).

Mask-to-polygon extraction, connected components and area measurement belong to the
pipeline owner. This service is what the **API** does with the facts a pipeline hands
back: it re-decides georeferencing from validation rather than trusting the pipeline's
claim, normalises units and artifact URLs, derives coarse location labels, and turns
structured evidence into a deterministic sentence when SatVLM composition is
unavailable.

Every spatial claim in an answer therefore traces to a mask, region, measurement or
validated metadata field, which is the rule in plan section 12.3.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.geospatial.crs import describe_crs, select_measurement_crs
from app.geospatial.overlap import relative_location
from app.schemas.common import AreaUnit, Task, Warning, WarningLevel
from app.schemas.evidence import Evidence, Overlay, Region


class EvidenceService:
    """Normalises and audits evidence coming from a pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def normalise(
        self,
        raw: Evidence | dict[str, Any] | None,
        *,
        analysis_id: str,
        georeferenced: bool,
        scene_bounds: list[list[float]] | None = None,
        measurement_crs: str | None = None,
        artifact_urls: dict[str, str] | None = None,
        synthetic: bool = False,
    ) -> Evidence:
        """Build an :class:`Evidence` block whose geographic claims the API controls.

        ``georeferenced`` always comes from validation, never from the pipeline: if a
        model asserts coordinates for an ungeoreferenced scene the claims are dropped
        and a warning replaces them.
        """
        payload: dict[str, Any] = raw.model_dump() if isinstance(raw, Evidence) else dict(raw or {})
        payload["georeferenced"] = bool(georeferenced)
        payload["synthetic"] = bool(synthetic) or bool(payload.get("synthetic"))

        if georeferenced:
            crs = payload.get("measurement_crs") or measurement_crs
            if not crs:
                selection = select_measurement_crs(payload.get("crs") or None, scene_bounds)
                crs = payload.get("measurement_crs") or selection.get("measurement_crs")
            payload["measurement_crs"] = crs
        else:
            payload["measurement_crs"] = None
            payload["geographic_coordinates"] = None

        if artifact_urls:
            payload = self._rewrite_artifact_urls(payload, artifact_urls)

        evidence = Evidence.model_validate(payload)
        evidence = self._complete_locations(evidence, scene_bounds=scene_bounds)
        return evidence

    # ---------------------------------------------------------------- URLs
    def _rewrite_artifact_urls(self, payload: dict[str, Any], artifact_urls: dict[str, str]) -> dict[str, Any]:
        """Replace pipeline-side artifact names with API-served URLs.

        A pipeline may name an artifact (``change-mask.png``) or reference it by
        ``artifact_id``; the client must only ever see an endpoint the API serves.
        """
        served_urls = set(artifact_urls.values())

        def resolve(value: Any) -> str | None:
            if not isinstance(value, dict):
                return None
            for key in ("artifact_id", "name", "filename", "url"):
                token = value.get(key)
                if token and str(token) in artifact_urls:
                    return artifact_urls[str(token)]
                if token and str(token) in served_urls:
                    return str(token)
            return None

        def warn_unregistered(kind: str) -> None:
            warnings = list(payload.get("warnings") or [])
            warnings.append(
                {
                    "code": "ARTIFACT_NOT_REGISTERED",
                    "level": "error",
                    "message": f"The pipeline referenced an unregistered {kind} artifact, so its URL was removed.",
                }
            )
            payload["warnings"] = warnings

        overlay = payload.get("overlay")
        if isinstance(overlay, dict):
            url = resolve(overlay)
            if url:
                overlay = {**overlay, "url": url}
                payload["overlay"] = overlay
            else:
                payload["overlay"] = None
                warn_unregistered("overlay")
        existing_geojson = payload.get("geojson_url")
        if existing_geojson and existing_geojson not in served_urls:
            if str(existing_geojson) in artifact_urls:
                payload["geojson_url"] = artifact_urls[str(existing_geojson)]
            else:
                payload["geojson_url"] = None
                warn_unregistered("GeoJSON")
        if not payload.get("geojson_url"):
            for key in ("geojson", "regions_geojson", "geojson_name", "geojson_artifact_id"):
                token = payload.get(key)
                if isinstance(token, str) and token in artifact_urls:
                    payload["geojson_url"] = artifact_urls[token]
                    break
        return payload

    # ---------------------------------------------------------- locations
    def _complete_locations(self, evidence: Evidence, *, scene_bounds: list[list[float]] | None) -> Evidence:
        """Fill coarse location labels from measured regions."""
        if not evidence.regions:
            return evidence
        sized = [region for region in evidence.regions if region.area_value is not None]
        largest = max(sized, key=lambda region: region.area_value or 0) if sized else evidence.regions[0]

        if largest.location is None and scene_bounds:
            largest.location = relative_location(largest.bounds, scene_bounds)
        for region in evidence.regions:
            if region.location is None and scene_bounds:
                region.location = relative_location(region.bounds, scene_bounds)

        updates: dict[str, Any] = {}
        if evidence.largest_region_location is None:
            updates["largest_region_location"] = largest.location
        if evidence.relative_location is None:
            updates["relative_location"] = largest.location
        if not evidence.region_count:
            updates["region_count"] = len(evidence.regions)
        if evidence.kind == "change" and evidence.changed_percentage is None and evidence.percentage is not None:
            updates["changed_percentage"] = evidence.percentage
        if evidence.kind == "change" and evidence.percentage is None and evidence.changed_percentage is not None:
            updates["percentage"] = evidence.changed_percentage
        if updates:
            evidence = evidence.model_copy(update=updates)
        return evidence

    # ---------------------------------------------------------- composing
    def summarise(self, evidence: Evidence | None, *, task: Task | str | None, question: str | None = None) -> str | None:
        """Deterministic, evidence-only sentence for template composition.

        Returns ``None`` when there are no facts to state, so the caller abstains
        instead of generating prose that nothing supports.
        """
        if evidence is None or evidence.kind in (None, "none"):
            return None

        wanted = str(getattr(task, "value", task))
        parts: list[str] = []

        if wanted == Task.BI_TEMPORAL_CHANGE.value or evidence.kind == "change":
            if evidence.region_count:
                parts.append(f"Change was detected in {evidence.region_count} region(s).")
            else:
                parts.append("No changed region was detected.")
            if evidence.area_value is not None:
                unit_label = _unit_label(evidence.area_unit)
                sentence = f"Changed area is {evidence.area_value:g} {unit_label}".strip()
                if evidence.measurement_crs:
                    sentence += f" measured in {evidence.measurement_crs}"
                if evidence.percentage is not None:
                    sentence += f" ({round(evidence.percentage, 2)}% of the scene)"
                parts.append(sentence + ".")
            elif evidence.percentage is not None:
                parts.append(f"About {round(evidence.percentage, 2)}% of the scene changed.")
            location = evidence.largest_region_location or evidence.relative_location
            if location:
                parts.append(f"The largest change is in the {location} of the scene.")

        elif wanted == Task.OPTICAL_SAR_LAND_COVER.value or evidence.kind == "land_cover":
            if not evidence.class_areas:
                return None
            labels = ", ".join(
                f"{_class_label(area.class_name)} {_format_amount(area.area_value)}{_unit_suffix(area.area_unit)}".strip()
                + (f" ({round(area.percentage, 1)}%)" if area.percentage is not None else "")
                for area in evidence.class_areas
            )
            parts.append(f"Segmentation of the optical and SAR scenes identified: {labels}.")
            location = evidence.largest_region_location or evidence.relative_location
            if location:
                parts.append(f"The dominant class is concentrated in the {location}.")

        else:
            # A scene description cannot be composed from measurements alone.
            return None

        if not evidence.georeferenced:
            parts.append(
                "Geographic coordinates and square-metre area are unavailable because the input "
                "is not georeferenced."
            )
        return " ".join(parts).strip() or None

    # ------------------------------------------------------------- helpers
    def check_overlay_shape(self, evidence: Evidence) -> list[Warning]:
        """Warn when an overlay cannot be drawn by Leaflet as-is (plan section 12.0)."""
        warnings: list[Warning] = []
        overlay: Overlay | None = evidence.overlay
        if overlay is None:
            warnings.append(
                Warning(
                    code="NO_OVERLAY_ARTIFACT",
                    level=WarningLevel.INFO,
                    message="No PNG overlay was produced, so the map cannot show the result spatially.",
                )
            )
            return warnings
        if overlay.format == "npy":
            warnings.append(
                Warning(
                    code="RAW_TENSOR_EXPOSED",
                    level=WarningLevel.ERROR,
                    message="Evidence references a raw tensor. The API must serve a PNG overlay plus "
                    "bounds or GeoJSON instead (plan section 12.0).",
                )
            )
        if evidence.georeferenced and overlay.bounds is None:
            warnings.append(
                Warning(
                    code="OVERLAY_BOUNDS_MISSING",
                    message="A georeferenced overlay was produced without bounds, so Leaflet cannot place it.",
                )
            )
        return warnings

    def pixel_region(self, region: dict[str, Any]) -> Region:
        """Convenience constructor the pipeline can be tested against."""
        return Region.model_validate(region)


def _unit_label(unit: AreaUnit | str | None) -> str:
    value = str(getattr(unit, "value", unit) or "")
    return {
        "m2": "m2",
        "km2": "km2",
        "ha": "hectares",
        "pixels": "pixels",
    }.get(value, value)


def _unit_suffix(unit: AreaUnit | str | None) -> str:
    label = _unit_label(unit)
    return f" {label}" if label else ""


def _format_amount(value: float | None) -> str:
    return f"{value:g}" if value is not None else "an unmeasured area of"


_CLASS_LABELS = {
    "built_up": "built-up",
    "builtup": "built-up",
    "water": "water",
    "vegetation": "vegetation",
    "other": "other",
}


def _class_label(name: str) -> str:
    return _CLASS_LABELS.get(str(name).lower().replace(" ", "_"), str(name).replace("_", " "))


_service: EvidenceService | None = None


def get_evidence_service(settings: Settings | None = None) -> EvidenceService:
    global _service
    if _service is None:
        _service = EvidenceService(settings)
    return _service
