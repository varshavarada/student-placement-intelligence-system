from pathlib import Path
import hashlib
import io
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


# ==================================================
# PROJECT IMPORTS
# ==================================================

from data_loader import load_data

from analytics import (
    get_branch_analysis,
    get_college_tier_analysis,
    get_project_analysis,
    get_salary_analysis,
    get_work_experience_analysis,
)

from data_validator import validate_placement_data

from schema_mapper import (
    apply_column_mapping,
    convert_salary_to_lpa,
    detect_column_mapping,
    get_optional_columns,
    get_required_columns,
)

from intelligence import (
    analyze_student,
    generate_recommendations,
    get_placed_student_benchmarks,
)


# ==================================================
# OPTIONAL ML IMPORT
# ==================================================

try:
    from ml_model import (
        get_or_train_model,
        predict_student_probability,
    )

    ML_AVAILABLE = True

except Exception:
    ML_AVAILABLE = False


# ==================================================
# OPTIONAL REPORT IMPORT
# ==================================================

try:
    from report_generator import generate_student_report

    REPORT_AVAILABLE = True

except Exception:
    REPORT_AVAILABLE = False


# ==================================================
# OPTIONAL CHATBOT IMPORT
# ==================================================

try:
    from chatbot import ask_placement_chatbot

    CHATBOT_AVAILABLE = True

except Exception:
    CHATBOT_AVAILABLE = False


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Placement Intelligence System",
    page_icon="🎓",
    layout="wide",
)


# ==================================================
# UI STYLING
# ==================================================

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 285px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 285px !important;
        }

        .block-container {
            padding-left: 2.5rem;
            padding-right: 2.5rem;
            padding-top: 2rem;
            max-width: 1400px;
        }

        section[data-testid="stSidebar"] label {
            font-size: 0.96rem;
        }

        section[data-testid="stSidebar"] hr {
            margin-top: 1.4rem;
            margin-bottom: 1.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# CONSTANTS
# ==================================================

NOT_MAPPED = "— Not mapped —"

INTELLIGENCE_REQUIRED_FEATURES = [
    "cgpa",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "mock_interview_score",
    "projects_count",
    "internships_count",
    "certifications_count",
    "github_repos",
]


DISPLAY_NAMES = {
    "student_id": "Student ID",
    "age": "Age",
    "gender": "Gender",
    "cgpa": "CGPA",
    "branch": "Branch",
    "college_tier": "College Tier",
    "internships_count": "Internships",
    "projects_count": "Projects",
    "certifications_count": "Certifications",
    "coding_skill_score": "Coding Skill",
    "aptitude_score": "Aptitude",
    "communication_skill_score": "Communication",
    "logical_reasoning_score": "Logical Reasoning",
    "hackathons_participated": "Hackathons",
    "github_repos": "GitHub Repositories",
    "linkedin_connections": "LinkedIn Connections",
    "mock_interview_score": "Mock Interview",
    "attendance_percentage": "Attendance",
    "backlogs": "Backlogs",
    "extracurricular_score": "Extracurricular",
    "leadership_score": "Leadership",
    "volunteer_experience": "Volunteer Experience",
    "sleep_hours": "Sleep Hours",
    "study_hours_per_day": "Study Hours / Day",
    "placement_status": "Placement Status",
    "salary_package_lpa": "Salary Package",
}


# ==================================================
# FILE HELPERS
# ==================================================

def read_uploaded_dataset(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(
            io.BytesIO(file_bytes)
        )

    if filename.endswith(".xlsx"):
        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    raise ValueError(
        "Only CSV and XLSX files are supported."
    )


def build_file_signature(uploaded_file):
    return hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()


# ==================================================
# ANALYTICS HELPERS
# ==================================================

def get_overall_metrics_safe(df):
    total_students = len(df)

    placed_students = int(
        (
            df["placement_status"]
            == "Placed"
        ).sum()
    )

    not_placed_students = (
        total_students
        - placed_students
    )

    placement_rate = (
        placed_students
        / total_students
        * 100
        if total_students > 0
        else 0
    )

    average_salary = None

    if "salary_package_lpa" in df.columns:

        placed_salary = df.loc[
            (
                df["placement_status"]
                == "Placed"
            ),
            "salary_package_lpa",
        ].dropna()

        if not placed_salary.empty:
            average_salary = (
                placed_salary.mean()
            )

    return {
        "total_students": total_students,
        "placed_students": placed_students,
        "not_placed_students": not_placed_students,
        "placement_rate": placement_rate,
        "average_salary": average_salary,
    }


def get_student_profile_columns(df):
    preferred_columns = [
        "student_id",
        "age",
        "gender",
        "branch",
        "college_tier",
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "logical_reasoning_score",
        "mock_interview_score",
        "projects_count",
        "internships_count",
        "certifications_count",
        "github_repos",
        "linkedin_connections",
        "attendance_percentage",
        "backlogs",
        "extracurricular_score",
        "leadership_score",
        "volunteer_experience",
        "placement_status",
        "salary_package_lpa",
    ]

    return [
        column
        for column in preferred_columns
        if column in df.columns
    ]


def format_profile_value(
    column,
    value,
):
    if pd.isna(value):
        return "Unavailable"

    if column == "salary_package_lpa":
        return f"{float(value):.2f} LPA"

    if column == "cgpa":
        return f"{float(value):.2f}"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


# ==================================================
# LOAD DEMO DATA
# ==================================================

try:
    demo_df = load_data()

except Exception as error:

    st.error(
        "Unable to load the demo dataset."
    )

    st.exception(error)

    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "active_df" not in st.session_state:
    st.session_state.active_df = (
        demo_df.copy()
    )

if "active_dataset_name" not in st.session_state:
    st.session_state.active_dataset_name = (
        "Demo Dataset"
    )

if "data_mode" not in st.session_state:
    st.session_state.data_mode = (
        "Demo Dataset"
    )

if "data_warnings" not in st.session_state:
    st.session_state.data_warnings = []

if "salary_conversion_message" not in st.session_state:
    st.session_state.salary_conversion_message = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_dataset_key" not in st.session_state:
    st.session_state.chat_dataset_key = None


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "🎓 Placement Intelligence"
)

st.sidebar.markdown(
    "### 📂 Data Source"
)

data_source = st.sidebar.radio(
    "Choose dataset",
    [
        "Demo Dataset",
        "Upload Data",
    ],
)


# ==================================================
# DEMO MODE
# ==================================================

if data_source == "Demo Dataset":

    st.session_state.active_df = (
        demo_df.copy()
    )

    st.session_state.active_dataset_name = (
        "Demo Dataset"
    )

    st.session_state.data_mode = (
        "Demo Dataset"
    )

    st.session_state.data_warnings = []

    st.session_state.salary_conversion_message = None


# ==================================================
# UPLOAD MODE
# ==================================================

else:

    st.sidebar.caption(
        "Upload institution placement data. "
        "Column names do not need to match "
        "the demo dataset."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Placement Data",
        type=[
            "csv",
            "xlsx",
        ],
    )

    if uploaded_file is not None:

        try:

            uploaded_df = read_uploaded_dataset(
                uploaded_file
            )

        except Exception as error:

            st.sidebar.error(
                "Unable to read the uploaded file."
            )

            st.sidebar.exception(error)

            uploaded_df = None


        if uploaded_df is not None:

            file_signature = build_file_signature(
                uploaded_file
            )

            st.sidebar.success(
                "File loaded successfully."
            )

            st.sidebar.caption(
                f"{len(uploaded_df):,} rows • "
                f"{len(uploaded_df.columns)} columns"
            )

            (
                automatic_mapping,
                unmapped_columns,
                missing_required_columns,
            ) = detect_column_mapping(
                uploaded_df
            )


            # ======================================
            # COLUMN MAPPING
            # ======================================

            st.sidebar.markdown(
                "### 🔗 Column Mapping"
            )

            st.sidebar.caption(
                "Suggested mappings are generated "
                "automatically. Review them before "
                "using the dataset."
            )

            uploaded_columns = list(
                uploaded_df.columns
            )

            mapping_options = [
                NOT_MAPPED,
                *uploaded_columns,
            ]

            selected_mapping = {}


            # ======================================
            # REQUIRED FIELDS
            # ======================================

            st.sidebar.markdown(
                "**Required Fields**"
            )

            for canonical_column in get_required_columns():

                suggested = automatic_mapping.get(
                    canonical_column
                )

                if suggested in uploaded_columns:

                    default_index = (
                        mapping_options.index(
                            suggested
                        )
                    )

                else:

                    default_index = 0


                selection = st.sidebar.selectbox(
                    (
                        f"{DISPLAY_NAMES.get(canonical_column, canonical_column)} *"
                    ),
                    mapping_options,
                    index=default_index,
                    key=(
                        f"{file_signature}_required_"
                        f"{canonical_column}"
                    ),
                )

                if selection != NOT_MAPPED:

                    selected_mapping[
                        canonical_column
                    ] = selection


            # ======================================
            # OPTIONAL FIELDS
            # ======================================

            with st.sidebar.expander(
                "Optional Fields"
            ):

                for canonical_column in get_optional_columns():

                    suggested = automatic_mapping.get(
                        canonical_column
                    )

                    if suggested in uploaded_columns:

                        default_index = (
                            mapping_options.index(
                                suggested
                            )
                        )

                    else:

                        default_index = 0


                    selection = st.selectbox(
                        DISPLAY_NAMES.get(
                            canonical_column,
                            canonical_column,
                        ),
                        mapping_options,
                        index=default_index,
                        key=(
                            f"{file_signature}_optional_"
                            f"{canonical_column}"
                        ),
                    )

                    if selection != NOT_MAPPED:

                        selected_mapping[
                            canonical_column
                        ] = selection


            # ======================================
            # SALARY UNIT
            # ======================================

            salary_unit = "LPA"

            salary_source_column = (
                selected_mapping.get(
                    "salary_package_lpa"
                )
            )

            if salary_source_column:

                st.sidebar.markdown(
                    "### 💰 Salary Format"
                )

                salary_unit = st.sidebar.selectbox(
                    "Salary Unit",
                    [
                        "LPA",
                        "Annual INR",
                        "Monthly INR",
                    ],
                    help=(
                        "Choose the unit used by "
                        "the uploaded salary or "
                        "package column. "
                        "The system converts salary "
                        "internally to LPA."
                    ),
                    key=(
                        f"salary_unit_"
                        f"{file_signature}"
                    ),
                )


                raw_salary = pd.to_numeric(
                    uploaded_df[
                        salary_source_column
                    ],
                    errors="coerce",
                ).dropna()


                if not raw_salary.empty:

                    median_salary = (
                        raw_salary.median()
                    )

                    if (
                        median_salary > 1000
                        and salary_unit == "LPA"
                    ):

                        st.sidebar.warning(
                            "These salary values look "
                            "large for LPA. If this "
                            "column stores annual salary "
                            "in rupees, select "
                            "'Annual INR'."
                        )


            # ======================================
            # MAPPING SUMMARY
            # ======================================

            with st.sidebar.expander(
                "Mapping Summary"
            ):

                if selected_mapping:

                    mapping_table = pd.DataFrame(
                        {
                            "System Field": [
                                DISPLAY_NAMES.get(
                                    key,
                                    key,
                                )
                                for key
                                in selected_mapping
                            ],

                            "Uploaded Column": list(
                                selected_mapping.values()
                            ),
                        }
                    )

                    st.dataframe(
                        mapping_table,
                        hide_index=True,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No fields are mapped yet."
                    )


            # ======================================
            # VALIDATE BUTTON
            # ======================================

            if st.sidebar.button(
                "Validate & Use Dataset",
                type="primary",
                use_container_width=True,
            ):

                required_columns = (
                    get_required_columns()
                )

                missing_mapping = [
                    column
                    for column
                    in required_columns
                    if column
                    not in selected_mapping
                ]


                if missing_mapping:

                    readable_missing = [
                        DISPLAY_NAMES.get(
                            column,
                            column,
                        )
                        for column
                        in missing_mapping
                    ]

                    st.sidebar.error(
                        "Please map all required "
                        "fields: "
                        + ", ".join(
                            readable_missing
                        )
                    )


                else:

                    mapped_sources = list(
                        selected_mapping.values()
                    )

                    duplicated_sources = {
                        source
                        for source
                        in mapped_sources
                        if (
                            mapped_sources.count(
                                source
                            )
                            > 1
                        )
                    }


                    if duplicated_sources:

                        st.sidebar.error(
                            "The same uploaded column "
                            "cannot be mapped to "
                            "multiple system fields: "
                            + ", ".join(
                                sorted(
                                    duplicated_sources
                                )
                            )
                        )


                    else:

                        mapped_df = apply_column_mapping(
                            uploaded_df,
                            selected_mapping,
                        )


                        # ==================================
                        # SALARY NORMALIZATION
                        # ==================================

                        mapped_df = convert_salary_to_lpa(
                            mapped_df,
                            salary_unit,
                        )


                        (
                            cleaned_df,
                            errors,
                            warnings,
                            summary,
                        ) = validate_placement_data(
                            mapped_df
                        )


                        if errors:

                            st.sidebar.error(
                                "Dataset validation failed."
                            )

                            for error in errors:

                                st.sidebar.error(
                                    error
                                )


                        else:

                            st.session_state.active_df = (
                                cleaned_df
                            )

                            st.session_state.active_dataset_name = (
                                uploaded_file.name
                            )

                            st.session_state.data_mode = (
                                "Uploaded Dataset"
                            )

                            st.session_state.data_warnings = (
                                warnings
                            )


                            if (
                                "salary_package_lpa"
                                in cleaned_df.columns
                                and salary_unit != "LPA"
                            ):

                                st.session_state.salary_conversion_message = (
                                    f"Salary values were "
                                    f"converted from "
                                    f"{salary_unit} to LPA."
                                )

                            else:

                                st.session_state.salary_conversion_message = (
                                    None
                                )


                            st.sidebar.success(
                                "Dataset validated successfully."
                            )


# ==================================================
# ACTIVE DATASET INFORMATION
# ==================================================

active_df = (
    st.session_state.active_df
)

active_dataset_name = (
    st.session_state.active_dataset_name
)

data_mode = (
    st.session_state.data_mode
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    "### Active Dataset"
)

st.sidebar.write(
    f"**{active_dataset_name}**"
)

st.sidebar.caption(
    f"{len(active_df):,} students"
)


if st.session_state.salary_conversion_message:

    st.sidebar.info(
        st.session_state.salary_conversion_message
    )


if st.session_state.data_warnings:

    with st.sidebar.expander(
        "Data Quality Warnings"
    ):

        for warning in (
            st.session_state.data_warnings
        ):

            st.warning(
                warning
            )


# ==================================================
# NAVIGATION
# ==================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "## Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Overview",
        "🎯 Student Insights",
        "💬 Placement Copilot",
    ],
)


# ==================================================
# HEADER
# ==================================================

st.title(
    "🎓 Student Placement Intelligence System"
)

st.caption(
    "Analyze placement outcomes, understand student "
    "readiness and explore institutional placement "
    "data through grounded AI insights."
)


if data_mode == "Demo Dataset":

    st.info(
        "🧪 Current Data Source: Demo Dataset • "
        "Synthetic placement data used for prototype testing."
    )

else:

    st.info(
        f"🏫 Current Data Source: "
        f"{active_dataset_name} • "
        f"{len(active_df):,} students"
    )


# ==================================================
# OVERVIEW PAGE
# ==================================================

if page == "📊 Overview":

    st.header(
        "📊 Placement Overview"
    )


    metrics = get_overall_metrics_safe(
        active_df
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Students",
        f"{metrics['total_students']:,}",
    )


    col2.metric(
        "Placed Students",
        f"{metrics['placed_students']:,}",
    )


    col3.metric(
        "Placement Rate",
        f"{metrics['placement_rate']:.2f}%",
    )


    if metrics["average_salary"] is not None:

        col4.metric(
            "Average Salary",
            f"{metrics['average_salary']:.2f} LPA",
        )

    else:

        col4.metric(
            "Average Salary",
            "Unavailable",
        )


    # ==============================================
    # AVAILABLE FIELDS
    # ==============================================

    with st.expander(
        "Available Institution Data"
    ):

        available_fields = [
            DISPLAY_NAMES.get(
                column,
                column,
            )
            for column
            in active_df.columns
        ]

        st.write(
            ", ".join(
                available_fields
            )
        )


    # ==============================================
    # PLACEMENT DISTRIBUTION
    # ==============================================

    st.subheader(
        "Placement Status Distribution"
    )

    placement_counts = (
        active_df[
            "placement_status"
        ]
        .value_counts()
        .reset_index()
    )

    placement_counts.columns = [
        "Placement Status",
        "Students",
    ]


    fig = px.pie(
        placement_counts,
        names="Placement Status",
        values="Students",
        hole=0.55,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ==============================================
    # BRANCH ANALYSIS
    # ==============================================

    st.subheader(
        "Branch-wise Placement Rate"
    )


    if "branch" in active_df.columns:

        try:

            branch_analysis = (
                get_branch_analysis(
                    active_df
                )
            )


            fig = px.bar(
                branch_analysis,
                x="branch",
                y="placement_rate",
                labels={
                    "branch": "Branch",
                    "placement_rate":
                        "Placement Rate (%)",
                },
            )

            fig.update_yaxes(
                range=[
                    0,
                    100,
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                "Branch analysis could not be calculated."
            )

            st.caption(
                str(error)
            )

    else:

        st.info(
            "Branch analysis is unavailable because "
            "the uploaded dataset does not contain "
            "a mapped branch field."
        )


    # ==============================================
    # COLLEGE TIER
    # ==============================================

    st.subheader(
        "College Tier Placement Rate"
    )


    if "college_tier" in active_df.columns:

        try:

            tier_analysis = (
                get_college_tier_analysis(
                    active_df
                )
            )


            fig = px.bar(
                tier_analysis,
                x="college_tier",
                y="placement_rate",
                labels={
                    "college_tier":
                        "College Tier",

                    "placement_rate":
                        "Placement Rate (%)",
                },
            )

            fig.update_yaxes(
                range=[
                    0,
                    100,
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                "College-tier analysis could "
                "not be calculated."
            )

            st.caption(
                str(error)
            )

    else:

        st.info(
            "College-tier analysis is unavailable "
            "because the uploaded dataset does not "
            "contain a mapped college-tier field."
        )


    # ==============================================
    # PREPARATION EXPERIENCE
    # ==============================================

    st.header(
        "Preparation Experience Trends"
    )


    # ==============================================
    # INTERNSHIPS
    # ==============================================

    st.subheader(
        "Internships vs Placement Rate"
    )


    if "internships_count" in active_df.columns:

        try:

            internship_analysis = (
                get_work_experience_analysis(
                    active_df
                )
            )


            reliable = internship_analysis[
                internship_analysis[
                    "reliable_group"
                ]
            ]


            if not reliable.empty:

                fig = px.line(
                    reliable,
                    x="internships_count",
                    y="placement_rate",
                    markers=True,
                    labels={
                        "internships_count":
                            "Internships",

                        "placement_rate":
                            "Placement Rate (%)",
                    },
                )

                fig.update_yaxes(
                    range=[
                        0,
                        100,
                    ]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


            st.caption(
                "Only groups with at least 100 students "
                "are treated as reliable. "
                "The relationship is associative, not causal."
            )


        except Exception as error:

            st.warning(
                "Internship analysis could "
                "not be calculated."
            )

            st.caption(
                str(error)
            )


    else:

        st.info(
            "Internship analysis is unavailable "
            "because internship data was not mapped."
        )


    # ==============================================
    # PROJECTS
    # ==============================================

    st.subheader(
        "Projects vs Placement Rate"
    )


    if "projects_count" in active_df.columns:

        try:

            project_analysis = (
                get_project_analysis(
                    active_df
                )
            )


            reliable_projects = (
                project_analysis[
                    project_analysis[
                        "reliable_group"
                    ]
                ]
            )


            if not reliable_projects.empty:

                fig = px.line(
                    reliable_projects,
                    x="projects_count",
                    y="placement_rate",
                    markers=True,
                    labels={
                        "projects_count":
                            "Projects",

                        "placement_rate":
                            "Placement Rate (%)",
                    },
                )

                fig.update_yaxes(
                    range=[
                        0,
                        100,
                    ]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


            st.caption(
                "Only groups with at least 100 students "
                "are treated as reliable. "
                "The relationship is associative, not causal."
            )


            low_sample_projects = (
                project_analysis[
                    ~project_analysis[
                        "reliable_group"
                    ]
                ]
            )


            if not low_sample_projects.empty:

                with st.expander(
                    "View Low-Sample Project Groups"
                ):

                    st.dataframe(
                        low_sample_projects,
                        hide_index=True,
                        use_container_width=True,
                    )


        except Exception as error:

            st.warning(
                "Project analysis could "
                "not be calculated."
            )

            st.caption(
                str(error)
            )


    else:

        st.info(
            "Project analysis is unavailable "
            "because project-count data was not mapped."
        )


    # ==============================================
    # SALARY ANALYTICS
    # ==============================================

    st.header(
        "Salary Analytics"
    )


    if "salary_package_lpa" in active_df.columns:

        try:

            salary_analysis = (
                get_salary_analysis(
                    active_df
                )
            )


            if salary_analysis is not None:

                (
                    salary_col1,
                    salary_col2,
                    salary_col3,
                    salary_col4,
                ) = st.columns(4)


                salary_col1.metric(
                    "Average Salary",
                    (
                        f"{salary_analysis['average_salary']:.2f} LPA"
                    ),
                )


                salary_col2.metric(
                    "Median Salary",
                    (
                        f"{salary_analysis['median_salary']:.2f} LPA"
                    ),
                )


                salary_col3.metric(
                    "Highest Salary",
                    (
                        f"{salary_analysis['highest_salary']:.2f} LPA"
                    ),
                )


                salary_col4.metric(
                    "75th Percentile",
                    (
                        f"{salary_analysis['q3_salary']:.2f} LPA"
                    ),
                )


                st.subheader(
                    "Salary Distribution"
                )


                placed_salary_df = (
                    active_df[
                        active_df[
                            "placement_status"
                        ]
                        == "Placed"
                    ][
                        [
                            "salary_package_lpa"
                        ]
                    ]
                    .dropna()
                )


                if not placed_salary_df.empty:

                    fig = px.histogram(
                        placed_salary_df,
                        x="salary_package_lpa",
                        nbins=30,
                        labels={
                            "salary_package_lpa":
                                "Salary Package (LPA)"
                        },
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No valid placed-student salary "
                        "values are available."
                    )


            else:

                st.info(
                    "Salary analytics are unavailable "
                    "because no valid salary values were found."
                )


        except Exception as error:

            st.warning(
                "Salary analytics could "
                "not be calculated."
            )

            st.caption(
                str(error)
            )


    else:

        st.info(
            "Salary analytics are unavailable because "
            "salary/package data was not mapped."
        )


# ==================================================
# STUDENT INTELLIGENCE PAGE
# ==================================================

elif page == "🎯 Student Insights":

    st.header("+🎯 Student Intelligence")


    if active_df.empty:

        st.warning(
            "The active dataset is empty."
        )

        st.stop()


    # ==============================================
    # STUDENT SELECTOR
    # ==============================================

    student_ids = (
        active_df[
            "student_id"
        ]
        .astype(str)
        .tolist()
    )


    selected_student_id = st.selectbox(
        "Select Student",
        student_ids,
    )


    student_matches = (
        active_df[
            active_df[
                "student_id"
            ].astype(str)
            == str(
                selected_student_id
            )
        ]
    )


    if student_matches.empty:

        st.error(
            "Student could not be found."
        )

        st.stop()


    student = (
        student_matches.iloc[0]
    )


    # ==============================================
    # STUDENT PROFILE
    # ==============================================

    st.subheader(
        "Student Profile"
    )


    profile_columns = (
        get_student_profile_columns(
            active_df
        )
    )


    profile_rows = []


    for column in profile_columns:

        profile_rows.append(
            {
                "Field":
                    DISPLAY_NAMES.get(
                        column,
                        column,
                    ),

                "Value":
                    format_profile_value(
                        column,
                        student[column],
                    ),
            }
        )


    st.dataframe(
        pd.DataFrame(
            profile_rows
        ),
        hide_index=True,
        use_container_width=True,
    )


    # ==============================================
    # CHECK INTELLIGENCE AVAILABILITY
    # ==============================================

    missing_intelligence_features = [
        feature
        for feature
        in INTELLIGENCE_REQUIRED_FEATURES
        if feature not in active_df.columns
    ]


    if missing_intelligence_features:

        st.info(
            "Complete student benchmark intelligence "
            "is unavailable for this dataset because "
            "some required analytical fields were not mapped."
        )


        readable_missing = [
            DISPLAY_NAMES.get(
                feature,
                feature,
            )
            for feature
            in missing_intelligence_features
        ]


        st.write(
            "**Missing fields:** "
            + ", ".join(
                readable_missing
            )
        )


        st.caption(
            "The dataset is still usable for cohort "
            "analytics and the student profile above. "
            "Unsupported intelligence modules are "
            "disabled instead of inventing missing information."
        )


    else:

        # ==========================================
        # BENCHMARKS
        # ==========================================

        benchmarks = (
            get_placed_student_benchmarks(
                active_df
            )
        )


        analysis = analyze_student(
            student,
            benchmarks,
        )


        # ==========================================
        # RECOMMENDATIONS
        # ==========================================

        recommendations = (
            generate_recommendations(
                analysis
            )
        )


        # ==========================================
        # COMPARISON
        # ==========================================

        st.subheader(
            "Benchmark Comparison"
        )


        comparison_rows = []


        for feature, comparison in (
            analysis[
                "comparisons"
            ].items()
        ):

            comparison_rows.append(
                {
                    "Feature":
                        DISPLAY_NAMES.get(
                            feature,
                            feature,
                        ),

                    "Student Value":
                        round(
                            comparison[
                                "student_value"
                            ],
                            2,
                        ),

                    "Placed Average":
                        round(
                            comparison[
                                "placed_average"
                            ],
                            2,
                        ),

                    "Difference":
                        round(
                            comparison[
                                "raw_difference"
                            ],
                            2,
                        ),
                }
            )


        st.dataframe(
            pd.DataFrame(
                comparison_rows
            ),
            hide_index=True,
            use_container_width=True,
        )


        # ==========================================
        # STRENGTHS / IMPROVEMENTS
        # ==========================================

        strength_col, improvement_col = (
            st.columns(2)
        )


        with strength_col:

            st.subheader(
                "Strengths"
            )


            if analysis["strengths"]:

                for item in analysis["strengths"]:

                    st.success(
                        item["feature"]
                    )

            else:

                st.info(
                    "No strong benchmark "
                    "advantages were detected."
                )


        with improvement_col:

            st.subheader(
                "Improvement Areas"
            )


            if analysis["improvement_areas"]:

                for item in (
                    analysis[
                        "improvement_areas"
                    ]
                ):

                    st.warning(
                        item["feature"]
                    )

            else:

                st.info(
                    "No major benchmark "
                    "gaps were detected."
                )


        # ==========================================
        # PREPARATION RECOMMENDATIONS
        # ==========================================

        st.subheader(
            "Preparation Recommendations"
        )


        if recommendations:

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )

        else:

            st.info(
                "No additional recommendations "
                "were generated."
            )


        # ==========================================
        # DEMO-ONLY ML + AI REPORT
        # ==========================================

        if data_mode == "Demo Dataset":

            prediction = None


            if ML_AVAILABLE:

                try:

                    pipeline = (
                        get_or_train_model(
                            active_df
                        )
                    )


                    prediction = (
                        predict_student_probability(
                            pipeline,
                            student,
                        )
                    )


                    st.subheader(
                        "ML Placement Estimate"
                    )


                    pred_col1, pred_col2 = (
                        st.columns(2)
                    )


                    pred_col1.metric(
                        "Model Classification",
                        prediction[
                            "prediction"
                        ],
                    )


                    pred_col2.metric(
                        "Placement Estimate",
                        (
                            f"{prediction['placement_probability']:.2f}%"
                        ),
                    )


                    st.caption(
                        "Prototype model estimate trained "
                        "on synthetic data. It is not a "
                        "guaranteed real-world placement probability."
                    )


                except Exception as error:

                    st.warning(
                        "ML estimate is temporarily unavailable."
                    )

                    st.caption(
                        str(error)
                    )


            if (
                REPORT_AVAILABLE
                and prediction is not None
            ):

                st.subheader(
                    "Placement Readiness Report"
                )


                try:

                    report_result = (
                        generate_student_report(
                            student,
                            analysis,
                            recommendations,
                            prediction,
                        )
                    )


                    if isinstance(
                        report_result,
                        dict,
                    ):

                        st.write(
                            report_result.get(
                                "report",
                                "",
                            )
                        )


                        source = (
                            report_result.get(
                                "source"
                            )
                        )


                        if source:

                            st.caption(
                                f"Report source: "
                                f"{source}"
                            )


                    else:

                        st.write(
                            report_result
                        )


                except Exception as error:

                    st.warning(
                        "The natural-language report "
                        "is temporarily unavailable."
                    )

                    st.caption(
                        str(error)
                    )


        else:

            st.info(
                "ML placement estimation and the current "
                "AI report are disabled for institution-uploaded "
                "datasets. The existing model was trained on "
                "the demo synthetic dataset and should not be "
                "applied to unrelated institutional data without "
                "compatible validation."
            )


# ==================================================
# PLACEMENT COPILOT PAGE
# ==================================================

elif page == "💬 Placement Copilot":

    st.header(
        "💬 Placement Copilot"
    )

    st.caption(
        "Ask questions about your active placement "
        "dataset and receive grounded, data-backed answers."
    )


    # ==============================================
    # DATASET-SPECIFIC CHAT SESSION
    # ==============================================

    current_chat_dataset_key = (
        f"{active_dataset_name}|"
        f"{len(active_df)}|"
        f"{','.join(active_df.columns)}"
    )


    if (
        st.session_state.chat_dataset_key
        != current_chat_dataset_key
    ):

        st.session_state.chat_history = []

        st.session_state.chat_dataset_key = (
            current_chat_dataset_key
        )


    # ==============================================
    # AVAILABILITY CHECK
    # ==============================================

    if not CHATBOT_AVAILABLE:

        st.error(
            "Placement Copilot could not be loaded."
        )

        st.caption(
            "Check that src/chatbot.py exists "
            "and that the required dependencies "
            "are installed."
        )

        st.stop()


    if active_df.empty:

        st.warning(
            "The active dataset is empty."
        )

        st.stop()


    # ==============================================
    # DATASET SUMMARY
    # ==============================================

    dataset_col1, dataset_col2 = (
        st.columns(2)
    )


    dataset_col1.metric(
        "Active Dataset",
        active_dataset_name,
    )


    dataset_col2.metric(
        "Students",
        f"{len(active_df):,}",
    )


    # ==============================================
    # AVAILABLE FIELDS
    # ==============================================

    available_fields = [
        DISPLAY_NAMES.get(
            column,
            column,
        )
        for column in active_df.columns
    ]


    with st.expander(
        "Fields available to the Copilot"
    ):

        st.write(
            ", ".join(
                available_fields
            )
        )


    # ==============================================
    # EXAMPLE QUESTIONS
    # ==============================================

    st.markdown(
        """
**Try asking**

- What is the placement rate?
- Which branch has the highest placement rate?
- Compare placed and not-placed students.
- Tell me about student 101.
"""
    )


    st.caption(
        "The Copilot answers from the active dataset. "
        "Unavailable information should be reported "
        "instead of being invented."
    )


    # ==============================================
    # CLEAR CHAT
    # ==============================================

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.chat_history = []

        st.rerun()


    # ==============================================
    # DISPLAY CHAT HISTORY
    # ==============================================

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


            if (
                message["role"] == "assistant"
                and message.get("source")
            ):

                source = message[
                    "source"
                ]


                if source == "python":

                    st.caption(
                        "✓ Calculated directly "
                        "from the active dataset"
                    )


                elif source == "groq":

                    st.caption(
                        "✓ Grounded response using "
                        "calculated dataset context"
                    )


                elif source == "fallback":

                    st.caption(
                        "Fallback response"
                    )


    # ==============================================
    # CHAT INPUT
    # ==============================================

    user_question = st.chat_input(
        "Ask about the active placement dataset..."
    )


    if user_question:

        # ------------------------------------------
        # STORE USER MESSAGE
        # ------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )


        # ------------------------------------------
        # DISPLAY USER MESSAGE
        # ------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_question
            )


        # ------------------------------------------
        # GENERATE RESPONSE
        # ------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing placement data..."
            ):

                try:

                    result = (
                        ask_placement_chatbot(
                            user_question,
                            active_df,
                        )
                    )


                    answer = result.get(
                        "answer",
                        (
                            "Unable to generate "
                            "a response."
                        ),
                    )


                    source = result.get(
                        "source",
                        "fallback",
                    )


                    st.markdown(
                        answer
                    )


                    if source == "python":

                        st.caption(
                            "✓ Calculated directly "
                            "from the active dataset"
                        )


                    elif source == "groq":

                        st.caption(
                            "✓ Grounded response using "
                            "calculated dataset context"
                        )


                    elif source == "fallback":

                        st.caption(
                            "Fallback response"
                        )


                    # ----------------------------------
                    # STORE ASSISTANT MESSAGE
                    # ----------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "source": source,
                        }
                    )


                except Exception as error:

                    error_message = (
                        "The Placement Copilot is "
                        "temporarily unavailable."
                    )


                    st.warning(
                        error_message
                    )

                    st.caption(
                        str(error)
                    )


                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "source": "fallback",
                        }
                    )


    # ==============================================
    # SCOPE NOTE
    # ==============================================

    st.markdown("---")

    st.caption(
        "Placement Copilot supports placement-data "
        "exploration and decision support. Observed "
        "dataset patterns are not causal evidence or "
        "automatic hiring decisions."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")


if data_mode == "Demo Dataset":

    st.caption(
        "Student Placement Intelligence System • "
        "Synthetic demo dataset"
    )

else:

    st.caption(
        "Student Placement Intelligence System • "
        "Institution dataset mode"
    )