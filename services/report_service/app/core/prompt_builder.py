from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.schemas.report import FindingsInput, PatientContext

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

_TEMPLATE_MAP = {
    "MRI": "mri_report.j2",
    "CXR": "cxr_report.j2",
    "DERM": "derm_report.j2",
}


def build_prompt(
    modality: str,
    findings: FindingsInput,
    patient_context: PatientContext | None,
) -> str:
    template_name = _TEMPLATE_MAP.get(modality.upper())
    if template_name is None:
        raise ValueError(f"Unsupported modality: {modality}")

    template = _env.get_template(template_name)
    return template.render(
        findings=findings.model_dump(),
        patient_context=patient_context.model_dump() if patient_context else None,
    )
