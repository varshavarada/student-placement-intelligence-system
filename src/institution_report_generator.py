import os
import math

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


PROFILE_LABELS = {
    "student_id": "Student ID",
    "age": "Age",
    "gender": "Gender",
    "branch": "Branch",
    "college_tier": "College Tier",
    "cgpa": "CGPA",
    "coding_skill_score": "Coding Skill",
    "aptitude_score": "Aptitude",
    "communication_skill_score": "Communication",
    "logical_reasoning_score": "Logical Reasoning",
    "mock_interview_score": "Mock Interview",
    "projects_count": "Projects",
    "internships_count": "Internships",
    "certifications_count": "Certifications",
    "github_repos": "GitHub Repositories",
    "placement_status": "Placement Status",
    "salary_package_lpa": "Salary Package (LPA)",
}


def _usable(value):
    if value is None:
        return False
    try:
        if bool(value != value):
            return False
    except Exception:
        pass
    return str(value).strip() != ""


def _format_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Unavailable"
        return f"{value:.2f}"
    return str(value)


def _profile_text(student):
    lines = []
    for field, label in PROFILE_LABELS.items():
        if field not in student.index:
            continue
        value = student.get(field)
        if not _usable(value):
            continue
        lines.append(f"- {label}: {_format_value(value)}")
    return "\n".join(lines) if lines else "- No additional profile fields available."


def _comparison_text(analysis):
    comparisons = analysis.get("comparisons", {})
    if not comparisons:
        return "- No valid placed-student benchmark comparisons are available."

    lines = []
    for feature, values in comparisons.items():
        label = PROFILE_LABELS.get(feature, feature)
        lines.append(
            f"- {label}: student={values['student_value']:.2f}, "
            f"placed_average={values['placed_average']:.2f}, "
            f"standardized_difference={values['normalized_difference']:.2f}"
        )
    return "\n".join(lines)


def _item_text(items, empty_message):
    if not items:
        return f"- {empty_message}"
    return "\n".join(
        f"- {item['feature']}: benchmark score {item['score']:.2f}"
        for item in items[:5]
    )


def _recommendation_text(recommendations):
    if not recommendations:
        return "- No additional preparation recommendation was generated from the available benchmark gaps."
    return "\n".join(f"- {item}" for item in recommendations)


def build_institution_report_prompt(
    student,
    analysis,
    recommendations,
    missing_features,
    dataset_name,
):
    missing_text = (
        ", ".join(missing_features)
        if missing_features
        else "None"
    )

    return f"""
Generate a professional placement-readiness report using ONLY the
verified information supplied below from an institution-uploaded dataset.

STRICT RULES:
1. Use only supplied facts and calculated benchmark comparisons.
2. Never invent missing student values, skills, achievements, companies,
   salaries, timelines, targets, or causal explanations.
3. This is institution-uploaded data. Do NOT call it synthetic unless the
   supplied facts explicitly say so.
4. Do NOT provide an ML placement probability or model classification.
   The demo-trained ML model is intentionally not used for this dataset.
5. If fields are missing, explicitly state that those areas could not be
   assessed from the uploaded dataset.
6. Benchmark differences are associative comparisons only, not causes of
   placement outcomes.
7. Do not recalculate or alter supplied numerical values.
8. Recommendations may only rephrase the system-generated recommendations.
9. Do not infer education stage, target career, company preference, or
   employment status beyond supplied fields.
10. Keep the tone professional and concise.

DATASET
- Active dataset: {dataset_name}

AVAILABLE STUDENT PROFILE
{_profile_text(student)}

PLACED-STUDENT BENCHMARK COMPARISONS
{_comparison_text(analysis)}

DETECTED STRENGTHS
{_item_text(analysis.get('strengths', []), 'No strong benchmark advantage was detected from the available fields.')}

PRIORITY IMPROVEMENT AREAS
{_item_text(analysis.get('improvement_areas', []), 'No major benchmark gap was detected from the available fields.')}

SYSTEM-GENERATED RECOMMENDATIONS
{_recommendation_text(recommendations)}

UNAVAILABLE ANALYTICAL FIELDS
- {missing_text}

Generate exactly these sections:
### Placement Readiness Summary
### Key Strengths
### Priority Improvement Areas
### Recommended Preparation Plan
### Data Availability Note
### Important Note

In the Important Note, state that this report uses the available mapped
institution data only, that missing fields were not estimated, and that no
synthetic-demo ML estimate was applied.
""".strip()


def generate_institution_fallback_report(
    student,
    analysis,
    recommendations,
    missing_features,
    dataset_name,
):
    strengths = analysis.get("strengths", [])
    improvements = analysis.get("improvement_areas", [])
    comparisons = analysis.get("comparisons", {})

    if strengths:
        strength_text = ", ".join(item["feature"] for item in strengths[:5])
    else:
        strength_text = "No strong benchmark advantage was detected from the available fields"

    if improvements:
        improvement_text = ", ".join(item["feature"] for item in improvements[:5])
    else:
        improvement_text = "No major benchmark gap was detected from the available fields"

    if recommendations:
        action_text = "\n".join(f"- {item}" for item in recommendations)
    else:
        action_text = "- No additional preparation recommendation was generated from the available benchmark gaps."

    if comparisons:
        comparison_summary = (
            f"{len(comparisons)} available analytical field(s) were compared "
            "with placed-student averages in the active dataset."
        )
    else:
        comparison_summary = (
            "No valid placed-student benchmark comparison could be calculated "
            "from the available analytical fields."
        )

    missing_text = (
        ", ".join(missing_features)
        if missing_features
        else "None"
    )

    student_id = student.get("student_id", "Unavailable")

    return f"""
### Placement Readiness Summary

Student ID **{student_id}** was reviewed using the available mapped fields
from **{dataset_name}**. {comparison_summary}

### Key Strengths

{strength_text}.

### Priority Improvement Areas

{improvement_text}.

### Recommended Preparation Plan

{action_text}

### Data Availability Note

Unavailable analytical fields: **{missing_text}**. These fields were not
estimated or filled with assumed values.

### Important Note

This report is based only on the available institution-uploaded data and
its placed-student benchmark comparisons. Missing fields were not inferred.
The synthetic-demo machine-learning model was not applied to this uploaded
dataset, so this report does not contain an ML placement probability.
""".strip()


def generate_institution_student_report(
    student,
    analysis,
    recommendations,
    missing_features,
    dataset_name="Uploaded Institution Dataset",
):
    prompt = build_institution_report_prompt(
        student,
        analysis,
        recommendations,
        missing_features,
        dataset_name,
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "report": generate_institution_fallback_report(
                student,
                analysis,
                recommendations,
                missing_features,
                dataset_name,
            ),
            "source": "fallback",
            "message": "GROQ_API_KEY was not found.",
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate grounded placement-readiness reports from "
                        "institution-uploaded data. Use only supplied facts, "
                        "never estimate missing values, and never invent an ML "
                        "prediction when one is not supplied."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1800,
        )

        if not completion.choices:
            raise ValueError("Groq returned no completion choices.")

        report = completion.choices[0].message.content
        if not report or not report.strip():
            raise ValueError("Groq returned an empty response.")

        return {
            "report": report.strip(),
            "source": "groq",
            "message": "Institution-safe report generated using Groq.",
        }

    except Exception as error:
        return {
            "report": generate_institution_fallback_report(
                student,
                analysis,
                recommendations,
                missing_features,
                dataset_name,
            ),
            "source": "fallback",
            "message": f"Groq unavailable. Fallback report used. Reason: {error}",
        }
