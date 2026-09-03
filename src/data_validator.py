import pandas as pd

from schema_mapper import (
    get_required_columns,
    normalize_dataset_values,
)


NUMERIC_COLUMNS = [
    "age",
    "cgpa",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs",
    "extracurricular_score",
    "leadership_score",
    "sleep_hours",
    "study_hours_per_day",
    "salary_package_lpa",
]
SCORE_COLUMNS = [
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "mock_interview_score",
    "extracurricular_score",
    "leadership_score",
]


COUNT_COLUMNS = [
    "internships_count",
    "projects_count",
    "certifications_count",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "backlogs",
]


def _convert_numeric_columns(df):
    cleaned_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column not in cleaned_df.columns:
            continue

        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    return cleaned_df


def _validate_required_columns(df):
    errors = []

    required_columns = get_required_columns()

    for column in required_columns:
        if column not in df.columns:
            errors.append(
                f"Required column missing: {column}"
            )

    return errors


def _validate_required_values(df):
    errors = []

    for column in get_required_columns():
        if column not in df.columns:
            continue

        missing_count = df[column].isna().sum()

        if missing_count > 0:
            errors.append(
                f"{column} contains {missing_count} missing values."
            )

    return errors


def _validate_placement_status(df):
    errors = []

    if "placement_status" not in df.columns:
        return errors

    allowed_values = {
        "Placed",
        "Not Placed",
    }

    invalid_values = (
        df.loc[
            ~df["placement_status"].isin(allowed_values)
            & df["placement_status"].notna(),
            "placement_status",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if invalid_values:
        preview = ", ".join(
            invalid_values[:5]
        )

        errors.append(
            "placement_status contains unsupported values: "
            f"{preview}"
        )

    return errors


def _validate_student_ids(df):
    errors = []
    warnings = []

    if "student_id" not in df.columns:
        return errors, warnings

    duplicate_count = df["student_id"].duplicated().sum()

    if duplicate_count > 0:
        errors.append(
            f"Found {duplicate_count} duplicate student IDs."
        )

    if df["student_id"].astype(str).str.strip().eq("").any():
        errors.append(
            "student_id contains empty values."
        )

    return errors, warnings


def _validate_ranges(df):
    errors = []
    warnings = []

    if "cgpa" in df.columns:
        invalid = df[
            df["cgpa"].notna()
            & ~df["cgpa"].between(0, 10)
        ]

        if not invalid.empty:
            errors.append(
                "CGPA must be between 0 and 10."
            )

    for column in SCORE_COLUMNS:
        if column not in df.columns:
            continue

        invalid = df[
            df[column].notna()
            & ~df[column].between(0, 100)
        ]

        if not invalid.empty:
            errors.append(
                f"{column} must be between 0 and 100."
            )

    if "attendance_percentage" in df.columns:
        invalid = df[
            df["attendance_percentage"].notna()
            & ~df["attendance_percentage"].between(0, 100)
        ]

        if not invalid.empty:
            errors.append(
                "attendance_percentage must be between 0 and 100."
            )

    if "age" in df.columns:
        unusual_age = df[
            df["age"].notna()
            & ~df["age"].between(15, 100)
        ]

        if not unusual_age.empty:
            warnings.append(
                "Some age values are outside the expected 15–100 range."
            )

    if "sleep_hours" in df.columns:
        invalid = df[
            df["sleep_hours"].notna()
            & ~df["sleep_hours"].between(0, 24)
        ]

        if not invalid.empty:
            warnings.append(
                "Some sleep_hours values are outside 0–24."
            )

    if "study_hours_per_day" in df.columns:
        invalid = df[
            df["study_hours_per_day"].notna()
            & ~df["study_hours_per_day"].between(0, 24)
        ]

        if not invalid.empty:
            warnings.append(
                "Some study_hours_per_day values are outside 0–24."
            )

    for column in COUNT_COLUMNS:
        if column not in df.columns:
            continue

        invalid = df[
            df[column].notna()
            & (df[column] < 0)
        ]

        if not invalid.empty:
            errors.append(
                f"{column} cannot contain negative values."
            )

    if "salary_package_lpa" in df.columns:
        invalid = df[
            df["salary_package_lpa"].notna()
            & (df["salary_package_lpa"] < 0)
        ]

        if not invalid.empty:
            errors.append(
                "salary_package_lpa cannot contain negative values."
            )

    return errors, warnings


def _validate_numeric_conversion(original_df, converted_df):
    warnings = []

    for column in NUMERIC_COLUMNS:
        if column not in original_df.columns:
            continue

        original_non_missing = original_df[column].notna()

        failed_conversion = (
            original_non_missing
            & converted_df[column].isna()
        )

        failed_count = failed_conversion.sum()

        if failed_count > 0:
            warnings.append(
                f"{column}: {failed_count} value(s) could not be "
                "converted to numeric format."
            )

    return warnings


def _validate_salary_consistency(df):
    warnings = []

    if (
        "placement_status" not in df.columns
        or "salary_package_lpa" not in df.columns
    ):
        return warnings

    placed_missing_salary = df[
        (df["placement_status"] == "Placed")
        & (
            df["salary_package_lpa"].isna()
            | (df["salary_package_lpa"] <= 0)
        )
    ]

    if not placed_missing_salary.empty:
        warnings.append(
            f"{len(placed_missing_salary)} placed student(s) "
            "have missing or zero salary values."
        )

    unplaced_with_salary = df[
        (df["placement_status"] == "Not Placed")
        & (df["salary_package_lpa"].fillna(0) > 0)
    ]

    if not unplaced_with_salary.empty:
        warnings.append(
            f"{len(unplaced_with_salary)} not-placed student(s) "
            "have positive salary values."
        )

    return warnings


def _generate_missing_feature_warnings(df):
    warnings = []

    useful_feature_groups = {
        "Academic performance": [
            "cgpa",
        ],

        "Technical skills": [
            "coding_skill_score",
        ],

        "Aptitude": [
            "aptitude_score",
            "logical_reasoning_score",
        ],

        "Communication / interview": [
            "communication_skill_score",
            "mock_interview_score",
        ],

        "Practical experience": [
            "projects_count",
            "internships_count",
        ],

        "Branch analysis": [
            "branch",
        ],

        "Salary analysis": [
            "salary_package_lpa",
        ],
    }

    for group_name, columns in useful_feature_groups.items():

        available = any(
            column in df.columns
            for column in columns
        )

        if not available:
            warnings.append(
                f"{group_name} data is unavailable. "
                "Related analysis will be skipped."
            )

    return warnings


def _build_summary(df):
    placed_students = 0
    not_placed_students = 0

    if "placement_status" in df.columns:
        placed_students = (
            df["placement_status"]
            .eq("Placed")
            .sum()
        )

        not_placed_students = (
            df["placement_status"]
            .eq("Not Placed")
            .sum()
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "placed_students": int(
            placed_students
        ),
        "not_placed_students": int(
            not_placed_students
        ),
        "available_columns": df.columns.tolist(),
    }


def validate_placement_data(df: pd.DataFrame):
    """
    Flexible validation for institution-uploaded placement data.

    The uploaded dataset is expected to have already passed through
    the schema-mapping layer.

    Returns:
        cleaned_df
        errors
        warnings
        summary
    """

    errors = []
    warnings = []

    if df is None:
        return (
            pd.DataFrame(),
            ["No dataset was provided."],
            [],
            {},
        )

    if df.empty:
        return (
            df.copy(),
            ["The uploaded dataset is empty."],
            [],
            {},
        )

    # Standardize supported categorical values first.
    normalized_df = normalize_dataset_values(
        df
    )

    # Convert available numeric columns safely.
    cleaned_df = _convert_numeric_columns(
        normalized_df
    )

    errors.extend(
        _validate_required_columns(
            cleaned_df
        )
    )

    errors.extend(
        _validate_required_values(
            cleaned_df
        )
    )

    errors.extend(
        _validate_placement_status(
            cleaned_df
        )
    )

    id_errors, id_warnings = (
        _validate_student_ids(
            cleaned_df
        )
    )

    errors.extend(id_errors)
    warnings.extend(id_warnings)

    range_errors, range_warnings = (
        _validate_ranges(
            cleaned_df
        )
    )

    errors.extend(range_errors)
    warnings.extend(range_warnings)

    warnings.extend(
        _validate_numeric_conversion(
            normalized_df,
            cleaned_df,
        )
    )

    warnings.extend(
        _validate_salary_consistency(
            cleaned_df
        )
    )

    warnings.extend(
        _generate_missing_feature_warnings(
            cleaned_df
        )
    )

    if len(cleaned_df) < 100:
        warnings.append(
            "The dataset contains fewer than 100 records. "
            "Analytics may be less reliable."
        )

    duplicate_rows = cleaned_df.duplicated().sum()

    if duplicate_rows > 0:
        warnings.append(
            f"The dataset contains {duplicate_rows} duplicate row(s)."
        )

    summary = _build_summary(
        cleaned_df
    )

    return (
        cleaned_df,
        errors,
        warnings,
        summary,
    )