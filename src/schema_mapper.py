import re

import pandas as pd


# ==================================================
# CANONICAL INTERNAL SCHEMA
# ==================================================

CANONICAL_COLUMNS = {
    "student_id": {
        "required": True,
        "aliases": [
            "student id",
            "student_id",
            "studentid",
            "student no",
            "student number",
            "register number",
            "register no",
            "reg no",
            "reg_no",
            "roll number",
            "roll no",
            "roll_no",
            "rollnumber",
            "sl no",
            "sl_no",
            "slno",
            "serial no",
            "serial number",
            "s no",
            "sno",
        ],
    },

    "placement_status": {
        "required": True,
        "aliases": [
            "placement status",
            "placement_status",
            "placementstatus",
            "status",
            "placed",
            "placed status",
            "selected",
            "selection status",
            "job status",
            "placement",
        ],
    },

    "age": {
        "required": False,
        "aliases": [
            "age",
            "student age",
        ],
    },

    "gender": {
        "required": False,
        "aliases": [
            "gender",
            "sex",
        ],
    },

    "cgpa": {
        "required": False,
        "aliases": [
            "cgpa",
            "cgpa score",
            "gpa",
            "academic cgpa",
            "academic score",
        ],
    },

    "branch": {
        "required": False,
        "aliases": [
            "branch",
            "department",
            "dept",
            "course",
            "program",
            "programme",
            "stream",
            "specialization",
            "specialisation",
        ],
    },

    "college_tier": {
        "required": False,
        "aliases": [
            "college tier",
            "college_tier",
            "tier",
            "institution tier",
        ],
    },

    "internships_count": {
        "required": False,
        "aliases": [
            "internships count",
            "internships_count",
            "internships",
            "internship count",
            "no of internships",
            "number of internships",
        ],
    },

    "projects_count": {
        "required": False,
        "aliases": [
            "projects count",
            "projects_count",
            "projects",
            "project count",
            "no of projects",
            "number of projects",
            "projects completed",
        ],
    },

    "certifications_count": {
        "required": False,
        "aliases": [
            "certifications count",
            "certifications_count",
            "certifications",
            "certification count",
            "certificates",
            "workshops certifications",
            "workshops and certifications",
        ],
    },

    "coding_skill_score": {
        "required": False,
        "aliases": [
            "coding skill score",
            "coding_skill_score",
            "coding score",
            "coding marks",
            "programming score",
            "programming marks",
            "coding skill",
            "technical score",
        ],
    },

    "aptitude_score": {
        "required": False,
        "aliases": [
            "aptitude score",
            "aptitude_score",
            "aptitude",
            "aptitude marks",
            "aptitude test score",
            "aptitudetestscore",
        ],
    },

    "communication_skill_score": {
        "required": False,
        "aliases": [
            "communication skill score",
            "communication_skill_score",
            "communication score",
            "communication",
            "communication marks",
            "communication skills",
            "soft skill rating",
            "softskillrating",
        ],
    },

    "logical_reasoning_score": {
        "required": False,
        "aliases": [
            "logical reasoning score",
            "logical_reasoning_score",
            "logical reasoning",
            "reasoning score",
            "logical score",
            "reasoning marks",
        ],
    },

    "hackathons_participated": {
        "required": False,
        "aliases": [
            "hackathons participated",
            "hackathons_participated",
            "hackathons",
            "hackathon count",
        ],
    },

    "github_repos": {
        "required": False,
        "aliases": [
            "github repos",
            "github_repos",
            "github repositories",
            "repositories",
            "repo count",
        ],
    },

    "linkedin_connections": {
        "required": False,
        "aliases": [
            "linkedin connections",
            "linkedin_connections",
            "linkedin",
        ],
    },

    "mock_interview_score": {
        "required": False,
        "aliases": [
            "mock interview score",
            "mock_interview_score",
            "mock interview",
            "interview score",
            "mock score",
        ],
    },

    "attendance_percentage": {
        "required": False,
        "aliases": [
            "attendance percentage",
            "attendance_percentage",
            "attendance",
            "attendance percent",
        ],
    },

    "backlogs": {
        "required": False,
        "aliases": [
            "backlogs",
            "backlog",
            "arrears",
            "arrear count",
        ],
    },

    "extracurricular_score": {
        "required": False,
        "aliases": [
            "extracurricular score",
            "extracurricular_score",
            "extracurricular",
            "extra curricular score",
            "extra curricular activities",
            "extracurricularactivities",
        ],
    },

    "leadership_score": {
        "required": False,
        "aliases": [
            "leadership score",
            "leadership_score",
            "leadership",
        ],
    },

    "volunteer_experience": {
        "required": False,
        "aliases": [
            "volunteer experience",
            "volunteer_experience",
            "volunteering",
            "volunteer",
        ],
    },

    "sleep_hours": {
        "required": False,
        "aliases": [
            "sleep hours",
            "sleep_hours",
            "sleep",
        ],
    },

    "study_hours_per_day": {
        "required": False,
        "aliases": [
            "study hours per day",
            "study_hours_per_day",
            "study hours",
            "daily study hours",
        ],
    },

    "salary_package_lpa": {
        "required": False,
        "aliases": [
            "salary package lpa",
            "salary_package_lpa",
            "salary",
            "package",
            "package lpa",
            "salary lpa",
            "ctc",
            "ctc lpa",
            "annual salary",
            "annual package",
        ],
    },
}


# ==================================================
# COLUMN NAME NORMALIZATION
# ==================================================

def normalize_column_name(column_name):
    column_name = str(
        column_name
    ).strip().lower()

    column_name = re.sub(
        r"[_\-]+",
        " ",
        column_name,
    )

    column_name = re.sub(
        r"[^a-z0-9 ]",
        "",
        column_name,
    )

    column_name = re.sub(
        r"\s+",
        " ",
        column_name,
    )

    return column_name.strip()


# ==================================================
# AUTOMATIC COLUMN DETECTION
# ==================================================

def detect_column_mapping(df):
    """
    Attempt to map uploaded dataset columns to
    the system's canonical internal schema.

    Returns:
        mapping
        unmapped_columns
        missing_required_columns
    """

    mapping = {}

    normalized_uploaded_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    used_uploaded_columns = set()

    for (
        canonical_name,
        metadata,
    ) in CANONICAL_COLUMNS.items():

        possible_names = [
            canonical_name,
            *metadata["aliases"],
        ]

        normalized_aliases = {
            normalize_column_name(
                name
            )
            for name in possible_names
        }

        matched_column = None

        for (
            normalized_name,
            original_column,
        ) in normalized_uploaded_columns.items():

            if (
                original_column
                in used_uploaded_columns
            ):
                continue

            if (
                normalized_name
                in normalized_aliases
            ):
                matched_column = (
                    original_column
                )
                break

        if matched_column is not None:

            mapping[
                canonical_name
            ] = matched_column

            used_uploaded_columns.add(
                matched_column
            )

    unmapped_columns = [
        column
        for column in df.columns
        if column
        not in used_uploaded_columns
    ]

    missing_required_columns = [
        canonical_name
        for (
            canonical_name,
            metadata,
        ) in CANONICAL_COLUMNS.items()
        if (
            metadata["required"]
            and canonical_name
            not in mapping
        )
    ]

    return (
        mapping,
        unmapped_columns,
        missing_required_columns,
    )


# ==================================================
# APPLY COLUMN MAPPING
# ==================================================

def apply_column_mapping(
    df,
    mapping,
):
    """
    Rename uploaded dataset columns into the
    system's canonical internal schema.

    Unknown/unmapped columns are ignored.
    """

    rename_mapping = {
        uploaded_column:
            canonical_column
        for (
            canonical_column,
            uploaded_column,
        ) in mapping.items()
        if (
            uploaded_column
            is not None
        )
    }

    mapped_df = df.rename(
        columns=rename_mapping
    ).copy()

    canonical_column_names = set(
        CANONICAL_COLUMNS.keys()
    )

    columns_to_keep = [
        column
        for column
        in mapped_df.columns
        if column
        in canonical_column_names
    ]

    return mapped_df[
        columns_to_keep
    ].copy()


# ==================================================
# PLACEMENT STATUS NORMALIZATION
# ==================================================

def normalize_placement_status(value):
    """
    Convert common placement status formats into:

        Placed
        Not Placed

    Unknown values are preserved so that the
    validator can report them instead of guessing.
    """

    if pd.isna(value):
        return value

    normalized = str(
        value
    ).strip().lower()

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        normalized,
    )

    placed_values = {
        "placed",
        "yes",
        "y",
        "1",
        "selected",
        "hired",
        "employed",
        "placement",
    }

    not_placed_values = {
        "notplaced",
        "no",
        "n",
        "0",
        "notselected",
        "unplaced",
        "nothired",
        "unemployed",
    }

    if compact in placed_values:
        return "Placed"

    if compact in not_placed_values:
        return "Not Placed"

    return value


# ==================================================
# COLLEGE TIER NORMALIZATION
# ==================================================

def normalize_college_tier(value):
    if pd.isna(value):
        return value

    normalized = str(
        value
    ).strip().lower()

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        normalized,
    )

    tier_mapping = {
        "1": "Tier 1",
        "tier1": "Tier 1",
        "t1": "Tier 1",

        "2": "Tier 2",
        "tier2": "Tier 2",
        "t2": "Tier 2",

        "3": "Tier 3",
        "tier3": "Tier 3",
        "t3": "Tier 3",
    }

    return tier_mapping.get(
        compact,
        value,
    )


# ==================================================
# YES / NO NORMALIZATION
# ==================================================

def normalize_yes_no(value):
    if pd.isna(value):
        return value

    normalized = str(
        value
    ).strip().lower()

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        normalized,
    )

    yes_values = {
        "yes",
        "y",
        "1",
        "true",
    }

    no_values = {
        "no",
        "n",
        "0",
        "false",
    }

    if compact in yes_values:
        return "Yes"

    if compact in no_values:
        return "No"

    return value


# ==================================================
# SALARY UNIT NORMALIZATION
# ==================================================

def convert_salary_to_lpa(
    df,
    salary_unit="LPA",
):
    """
    Convert salary values into the system's
    standard internal unit: LPA.

    Supported input units:
        LPA
        Annual INR
        Monthly INR
    """

    converted_df = df.copy()

    if (
        "salary_package_lpa"
        not in converted_df.columns
    ):
        return converted_df

    salary = pd.to_numeric(
        converted_df[
            "salary_package_lpa"
        ],
        errors="coerce",
    )

    if salary_unit == "Annual INR":

        salary = (
            salary / 100000
        )

    elif salary_unit == "Monthly INR":

        salary = (
            salary
            * 12
            / 100000
        )

    elif salary_unit == "LPA":

        pass

    else:

        raise ValueError(
            "Unsupported salary unit. "
            "Use LPA, Annual INR, "
            "or Monthly INR."
        )

    converted_df[
        "salary_package_lpa"
    ] = salary

    return converted_df


# ==================================================
# COMPLETE VALUE NORMALIZATION
# ==================================================

def normalize_dataset_values(df):
    """
    Normalize known categorical/value formats.

    Numerical conversion and validation are
    handled separately by data_validator.py.
    """

    normalized_df = df.copy()

    if (
        "placement_status"
        in normalized_df.columns
    ):

        normalized_df[
            "placement_status"
        ] = normalized_df[
            "placement_status"
        ].apply(
            normalize_placement_status
        )

    if (
        "college_tier"
        in normalized_df.columns
    ):

        normalized_df[
            "college_tier"
        ] = normalized_df[
            "college_tier"
        ].apply(
            normalize_college_tier
        )

    if (
        "volunteer_experience"
        in normalized_df.columns
    ):

        normalized_df[
            "volunteer_experience"
        ] = normalized_df[
            "volunteer_experience"
        ].apply(
            normalize_yes_no
        )

    return normalized_df


# ==================================================
# REQUIRED / OPTIONAL COLUMN HELPERS
# ==================================================

def get_required_columns():
    return [
        column
        for (
            column,
            metadata,
        ) in CANONICAL_COLUMNS.items()
        if metadata["required"]
    ]


def get_optional_columns():
    return [
        column
        for (
            column,
            metadata,
        ) in CANONICAL_COLUMNS.items()
        if not metadata["required"]
    ]


# ==================================================
# AVAILABLE / MISSING FEATURE HELPERS
# ==================================================

def get_available_features(df):
    return [
        column
        for column
        in CANONICAL_COLUMNS
        if column in df.columns
    ]


def get_missing_features(df):
    return [
        column
        for column
        in CANONICAL_COLUMNS
        if column not in df.columns
    ]