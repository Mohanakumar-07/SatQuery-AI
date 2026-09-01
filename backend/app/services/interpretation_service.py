"""Input interpretation: infer single-image, bi-temporal or optical-SAR (section 7.1).

The user supplies files and a question, never a model choice (section 9). This service
decides the input *mode* and each file's *role* from file count, probed metadata,
sensors, dates, modality evidence and query intent - and returns
``missing_fields`` whenever the answer would be a guess. Ambiguity is escalated as
``needs_clarification`` rather than resolved by assumption, which is why certainty is
tracked explicitly instead of being implied by a non-empty result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.core.config import Settings, get_settings
from app.db.models import Upload
from app.geospatial.modality import ModalityGuess, infer_modality
from app.schemas.analyses import AnalysisHints
from app.schemas.common import ClarificationField, FileRole, InputType, Modality
from app.services.query_parser import ParsedQuery


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


@dataclass
class InterpretationResult:
    input_type: InputType | None = None
    modalities: list[Modality] = field(default_factory=list)
    file_roles: dict[str, FileRole] = field(default_factory=dict)
    certainty: float = 0.0
    rationale: list[str] = field(default_factory=list)
    missing_fields: list[ClarificationField] = field(default_factory=list)
    clarification_question: str | None = None
    allowed_roles: list[FileRole] = field(default_factory=list)
    dates: dict[str, str] = field(default_factory=dict)
    modality_evidence: dict[str, list[str]] = field(default_factory=dict)
    #: True when a change question was asked, regardless of the resolved input type.
    change_requested: bool = False

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_fields)

    @property
    def georeferenced(self) -> bool:
        return bool(self.dates) or True  # placeholder, replaced by validation

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_input_type": self.input_type.value if self.input_type else None,
            "detected_modalities": [m.value for m in self.modalities],
            "file_roles": {key: role.value for key, role in self.file_roles.items()},
            "certainty": round(self.certainty, 3),
            "rationale": self.rationale,
            "missing_fields": [f.value for f in self.missing_fields],
            "question": self.clarification_question,
            "allowed_roles": [role.value for role in self.allowed_roles],
            "dates": self.dates,
            "modality_evidence": self.modality_evidence,
        }


def _modality_for(upload: Upload, *, sensor_hint: str | None) -> ModalityGuess:
    probe = upload.probe or {}
    return infer_modality(
        sensor=sensor_hint or upload.sensor,
        band_names=probe.get("band_names") or [],
        declared=upload.modality,
        media_kind=upload.media_kind,
        filename=upload.original_filename,
    )


def _ordered_by_date(pairs: list[tuple[Upload, date | None]]) -> list[tuple[Upload, date | None]] | None:
    """Sort uploads by acquisition date, or None when any date is missing."""
    if any(moment is None for _, moment in pairs):
        return None
    return sorted(pairs, key=lambda item: item[1])  # type: ignore[arg-type,return-value]


def interpret_inputs(
    uploads: list[Upload],
    *,
    hints: AnalysisHints | None = None,
    parsed: ParsedQuery | None = None,
    settings: Settings | None = None,
) -> InterpretationResult:
    """Infer the input mode and per-file roles for a set of validated uploads."""
    settings = settings or get_settings()
    hints = hints or AnalysisHints()
    parsed = parsed or ParsedQuery(question="", normalized="")
    result = InterpretationResult(change_requested=parsed.asks_about_change)

    if not uploads:
        result.missing_fields = [ClarificationField.FILE_ROLES]
        result.clarification_question = "Upload at least one image before asking a question."
        return result

    if len(uploads) > settings.max_files_per_analysis:
        result.rationale.append(
            f"{len(uploads)} files submitted but the MVP supports at most {settings.max_files_per_analysis}."
        )
        result.certainty = 0.0
        return result

    sensor_hints = list(hints.sensor_names or [])
    guesses: dict[str, ModalityGuess] = {}
    for index, upload in enumerate(uploads):
        hint = sensor_hints[index] if index < len(sensor_hints) else None
        guesses[upload.id] = _modality_for(upload, sensor_hint=hint)
        result.modalities.append(guesses[upload.id].modality)
        result.modality_evidence[upload.id] = guesses[upload.id].evidence
        upload_date = _parse_date(upload.acquisition_date)
        if upload_date:
            result.dates[upload.id] = upload_date.isoformat()

    supplied_roles = _apply_supplied_roles(uploads, hints, result)
    if supplied_roles:
        return result

    if len(uploads) == 1:
        _interpret_single(uploads[0], guesses[uploads[0].id], result, change_requested=parsed.asks_about_change)
        return result

    _interpret_pair(
        uploads,
        guesses,
        result,
        parsed=parsed,
        hints=hints,
        min_overlap_percent=settings.min_overlap_percent,
    )
    return result


def _apply_supplied_roles(uploads: list[Upload], hints: AnalysisHints, result: InterpretationResult) -> bool:
    """Honour explicit roles the client supplied when resolving a clarification."""
    if not hints.file_roles:
        return False
    known = {upload.id for upload in uploads}
    invalid = {key for key in hints.file_roles if key not in known}
    if invalid:
        result.rationale.append(f"Ignored roles for unknown uploads: {sorted(invalid)}.")
        return False

    roles = dict(hints.file_roles)
    wanted = {FileRole.BEFORE.value, FileRole.AFTER.value} | {FileRole.OPTICAL.value, FileRole.SAR.value}
    values = {role.value for role in roles.values()}
    if values and not values <= wanted:
        result.rationale.append(f"Roles {sorted(values - wanted)} are not part of the MVP vocabulary.")
        return False

    result.file_roles = roles
    result.certainty = 0.98
    result.rationale.append("File roles were supplied explicitly by the client.")
    if values & {FileRole.BEFORE.value, FileRole.AFTER.value}:
        result.input_type = InputType.BI_TEMPORAL
    elif values & {FileRole.OPTICAL.value, FileRole.SAR.value}:
        result.input_type = InputType.OPTICAL_SAR
    elif FileRole.SINGLE in values:
        result.input_type = InputType.SINGLE_IMAGE
    return result.input_type is not None


def _interpret_single(
    upload: Upload,
    guess: ModalityGuess,
    result: InterpretationResult,
    *,
    change_requested: bool,
) -> None:
    result.input_type = InputType.SINGLE_IMAGE
    result.file_roles[upload.id] = FileRole.SINGLE
    result.rationale.append("One file submitted, so the input is a single scene.")
    if guess.modality is Modality.SAR:
        result.rationale.append(
            "Single SAR scene: the documented VV/VH visualisation adapter is required (plan section 18)."
        )
    result.certainty = 0.95
    if change_requested:
        result.missing_fields = [ClarificationField.FILE_ROLES]
        result.clarification_question = (
            "A change question needs a before and an after image, but only one file was provided. "
            "Upload the second image."
        )
        result.allowed_roles = [FileRole.BEFORE, FileRole.AFTER]
        result.certainty = 0.4


def _interpret_pair(
    uploads: list[Upload],
    guesses: dict[str, ModalityGuess],
    result: InterpretationResult,
    *,
    parsed: ParsedQuery,
    hints: AnalysisHints,
    min_overlap_percent: float,
) -> None:
    first, second = uploads
    modality_first = guesses[first.id].modality
    modality_second = guesses[second.id].modality
    pair = [(upload, _parse_date(upload.acquisition_date)) for upload in uploads]

    undetermined = [upload.id for upload in uploads if not guesses[upload.id].decided]
    if undetermined:
        result.rationale.append(
            "Modality could not be determined for "
            f"{', '.join(sorted(undetermined))}; band names, sensor metadata or a declaration are needed."
        )
        result.missing_fields.append(ClarificationField.MODALITY)
        result.clarification_question = (
            "I cannot tell whether these two files are optical, SAR, or two dates of the same sensor. "
            "Which sensor or modality is each file?"
        )
        result.allowed_roles = [FileRole.BEFORE, FileRole.AFTER, FileRole.OPTICAL, FileRole.SAR]
        result.certainty = 0.3
        return

    modalities = {modality_first, modality_second}

    if modalities == {Modality.OPTICAL, Modality.SAR}:
        if parsed.asks_about_change and all(moment for _, moment in pair) and pair[0][1] != pair[1][1]:
            # Different dates across different sensors: the question is temporal, but an
            # optical-SAR pair is not a validated temporal pair.
            result.rationale.append(
                "A change question over an optical-SAR pair is not a supported temporal workflow: "
                "the difference may be sensor behaviour rather than ground change."
            )
            result.missing_fields.append(ClarificationField.MODALITY)
            result.clarification_question = (
                "These two files are different sensors (optical and SAR) taken at different dates. "
                "SatQuery can segment them as an optical-SAR pair, or compare two dates of one "
                "sensor. Which do you want?"
            )
            result.allowed_roles = [FileRole.OPTICAL, FileRole.SAR]
            result.certainty = 0.35
            return

        result.input_type = InputType.OPTICAL_SAR
        for upload, guess in ((first, modality_first), (second, modality_second)):
            result.file_roles[upload.id] = FileRole.OPTICAL if guess is Modality.OPTICAL else FileRole.SAR
        result.rationale.append("One optical and one SAR scene describe a co-registered fusion pair.")
        result.certainty = 0.9
        return

    # Same modality twice: a temporal pair if the dates differ.
    dated = _ordered_by_date(pair)
    if dated is not None:
        earlier, later = dated
        if earlier[1] == later[1]:
            result.rationale.append("Both files share the same acquisition date, so they are not a temporal pair.")
            result.missing_fields.extend([ClarificationField.BEFORE_DATE, ClarificationField.AFTER_DATE])
            result.clarification_question = (
                "Both images are dated the same day. What are the earlier and later acquisition dates?"
            )
            result.allowed_roles = [FileRole.BEFORE, FileRole.AFTER]
            result.certainty = 0.35
            return
        result.input_type = InputType.BI_TEMPORAL
        result.file_roles[earlier[0].id] = FileRole.BEFORE
        result.file_roles[later[0].id] = FileRole.AFTER
        result.certainty = 0.92
        result.rationale.append(
            f"Two {modality_first.value} scenes with different acquisition dates "
            f"({earlier[1].isoformat()} then {later[1].isoformat()}); ordering came from the dates, "
            "not from upload order."
        )
        if hints.before_date and hints.after_date:
            hint_earlier = _parse_date(hints.before_date)
            hint_later = _parse_date(hints.after_date)
            if hint_earlier and hint_later and hint_earlier > hint_later:
                result.rationale.append("Client hint dates were reversed; file order was taken from the metadata dates.")
        return

    # Dates unavailable. A change question with two same-sensor files is still very
    # likely temporal, but which file is earlier cannot be assumed.
    if parsed.asks_about_change or (hints.before_date and hints.after_date):
        result.input_type = InputType.BI_TEMPORAL
        result.missing_fields.append(ClarificationField.FILE_ROLES)
        result.clarification_question = "Which file is the earlier image?"
        result.allowed_roles = [FileRole.BEFORE, FileRole.AFTER]
        result.certainty = 0.45
        if hints.before_date and hints.after_date:
            hint_earlier = _parse_date(hints.before_date)
            hint_later = _parse_date(hints.after_date)
            if hint_earlier and hint_later and hint_earlier < hint_later:
                # The client named the dates and the upload order: honour that order,
                # recording that it came from submission order rather than metadata.
                result.file_roles[first.id] = FileRole.BEFORE
                result.file_roles[second.id] = FileRole.AFTER
                result.missing_fields = []
                result.certainty = 0.7
                result.rationale.remove("Which file is the earlier image?")
                result.rationale.append(
                    "No acquisition dates in the files; temporal order taken from the hint dates "
                    "combined with upload order."
                )
        return

    result.input_type = InputType.BI_TEMPORAL
    result.missing_fields.append(ClarificationField.FILE_ROLES)
    result.clarification_question = "Which file is the earlier image, and what is the question about them?"
    result.allowed_roles = [FileRole.BEFORE, FileRole.AFTER]
    result.certainty = 0.3
    result.rationale.append(
        "Two same-sensor files with no dates and no change wording: the relationship is ambiguous."
    )
