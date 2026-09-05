import math
import pandas as pd


FEATURE_LABELS = {
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
}

BENCHMARK_COLUMNS = list(FEATURE_LABELS.keys())

STRENGTH_THRESHOLD = 0.25
IMPROVEMENT_THRESHOLD = -0.25


def _to_number(value):
    """Return a finite float or None without inventing values."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def get_placed_student_benchmarks(df):

    if "placement_status" not in df.columns:
        return {}

    placed_students = df[
        df["placement_status"].astype(str).str.strip().str.lower() == "placed"
    ].copy()

    if placed_students.empty:
        return {}

    benchmarks = {}

    for column in BENCHMARK_COLUMNS:

        if column not in placed_students.columns:
            continue

        numeric_values = pd.to_numeric(
            placed_students[column],
            errors="coerce",
        ).dropna()

        if numeric_values.empty:
            continue

        benchmarks[column] = {
            "mean": numeric_values.mean(),
            "std": numeric_values.std(),
        }

    return benchmarks


def analyze_student(student, benchmarks):
    """Analyze only student fields for which a valid benchmark exists."""
    analysis = {
        "strengths": [],
        "improvement_areas": [],
        "comparisons": {},
    }

    for feature, benchmark_data in benchmarks.items():
        if feature not in student.index:
            continue

        student_value = _to_number(student.get(feature))
        benchmark_mean = _to_number(benchmark_data.get("mean"))
        benchmark_std = _to_number(benchmark_data.get("std"))

        if student_value is None or benchmark_mean is None:
            continue

        raw_difference = student_value - benchmark_mean

        if benchmark_std is not None and benchmark_std > 0:
            normalized_difference = raw_difference / benchmark_std
        else:
            normalized_difference = 0.0

        analysis["comparisons"][feature] = {
            "student_value": student_value,
            "placed_average": benchmark_mean,
            "raw_difference": raw_difference,
            "normalized_difference": normalized_difference,
        }

        item = {
            "feature": FEATURE_LABELS.get(feature, feature),
            "raw_difference": raw_difference,
            "score": abs(normalized_difference),
        }

        if normalized_difference >= STRENGTH_THRESHOLD:
            analysis["strengths"].append(item)
        elif normalized_difference <= IMPROVEMENT_THRESHOLD:
            analysis["improvement_areas"].append(item)

    analysis["strengths"].sort(
        key=lambda item: item["score"],
        reverse=True,
    )
    analysis["improvement_areas"].sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return analysis


def generate_recommendations(analysis):
    recommendations = []
    weak_areas = analysis.get("improvement_areas", [])

    recommendation_map = {
        "Coding Skill": (
            "Improve coding ability through regular problem solving "
            "and practical development."
        ),
        "Aptitude": (
            "Prioritize quantitative aptitude and placement-test practice."
        ),
        "Communication": (
            "Practice structured speaking, group discussions and "
            "interview communication."
        ),
        "Logical Reasoning": (
            "Strengthen logical reasoning through timed analytical "
            "problem-solving practice."
        ),
        "Mock Interview": (
            "Increase mock interview practice and review performance "
            "after every session."
        ),
        "Projects": (
            "Build additional practical projects and document them "
            "clearly in your portfolio."
        ),
        "Internships": (
            "Gain practical exposure through internships or "
            "industry-oriented projects."
        ),
        "GitHub Repositories": (
            "Publish and improve your strongest existing projects on GitHub "
            "with clear README files, documentation and regular version control."
        ),
        "Certifications": (
            "Complete a relevant certification that supports your current "
            "technical profile and placement preparation."
        ),
        "CGPA": (
            "Improve academic performance where possible while maintaining "
            "placement preparation."
        ),
    }

    for item in weak_areas[:5]:
        feature = item.get("feature")
        recommendation = recommendation_map.get(feature)
        if recommendation:
            recommendations.append(recommendation)

    return recommendations
