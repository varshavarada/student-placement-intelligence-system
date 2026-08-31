from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from data_loader import load_data

from analytics import (
    get_overall_metrics,
    get_branch_analysis,
    get_college_tier_analysis,
    get_salary_analysis,
    get_work_experience_analysis,
    get_project_analysis,
)

from intelligence import (
    get_placed_student_benchmarks,
    analyze_student,
    generate_recommendations,
)

from ml_model import (
    load_model,
    predict_student_probability,
)

from report_generator import generate_student_report


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Placement Intelligence System",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------

@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return load_model()


df = get_data()
model = get_model()


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title(
    "🎓 Student Placement Intelligence System"
)

st.caption(
    "Placement analytics, student benchmarking, "
    "personalized preparation insights, ML-based estimation "
    "and AI-generated placement readiness reports."
)


# --------------------------------------------------
# DATA DISCLAIMER
# --------------------------------------------------

st.info(
    "This prototype currently uses synthetic student placement data. "
    "Analytics, recommendations and model outputs should be interpreted "
    "as decision-support results rather than real-world guarantees."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Student Intelligence",
    ],
)


# ==================================================
# OVERVIEW PAGE
# ==================================================

if page == "Overview":

    metrics = get_overall_metrics(
        df
    )

    st.header(
        "Placement Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

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

    col4.metric(
        "Average Salary",
        f"{metrics['average_salary']:.2f} LPA",
    )


    # --------------------------------------------------
    # PLACEMENT STATUS DISTRIBUTION
    # --------------------------------------------------

    st.subheader(
        "Placement Status Distribution"
    )

    placement_counts = (
        df["placement_status"]
        .value_counts()
        .reset_index()
    )

    placement_counts.columns = [
        "Placement Status",
        "Students",
    ]

    fig_status = px.pie(
        placement_counts,
        names="Placement Status",
        values="Students",
        hole=0.5,
    )

    fig_status.update_layout(
        legend_title_text="Status"
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True,
    )


    # --------------------------------------------------
    # BRANCH + COLLEGE TIER ANALYSIS
    # --------------------------------------------------

    left, right = (
        st.columns(2)
    )


    with left:

        st.subheader(
            "Branch-wise Placement Rate"
        )

        branch_analysis = (
            get_branch_analysis(df)
            .reset_index()
        )

        branch_analysis.columns = [
            "Branch",
            "Placement Rate",
        ]

        fig_branch = px.bar(
            branch_analysis,
            x="Placement Rate",
            y="Branch",
            orientation="h",
            text_auto=".2f",
        )

        fig_branch.update_layout(
            xaxis_title="Placement Rate (%)",
            yaxis_title="",
            xaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_branch,
            use_container_width=True,
        )

        st.caption(
            "Branch placement rates are relatively close in this "
            "synthetic dataset, so small differences should not "
            "be over-interpreted."
        )


    with right:

        st.subheader(
            "College Tier Placement Rate"
        )

        tier_analysis = (
            get_college_tier_analysis(df)
            .reset_index()
        )

        tier_analysis.columns = [
            "College Tier",
            "Placement Rate",
        ]

        fig_tier = px.bar(
            tier_analysis,
            x="College Tier",
            y="Placement Rate",
            text_auto=".2f",
        )

        fig_tier.update_layout(
            yaxis_title="Placement Rate (%)",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_tier,
            use_container_width=True,
        )

        st.caption(
            "College-tier placement rates are also very similar "
            "in this dataset."
        )


    # --------------------------------------------------
    # EXPERIENCE + PROJECT TREND ANALYSIS
    # --------------------------------------------------

    st.header(
        "Preparation Experience Trends"
    )

    trend_left, trend_right = (
        st.columns(2)
    )


    with trend_left:

        st.subheader(
            "Internships vs Placement Rate"
        )

        internship_analysis = (
            get_work_experience_analysis(
                df
            )
        )

        reliable_internships = (
            internship_analysis[
                internship_analysis[
                    "reliable_group"
                ]
            ]
            .copy()
        )

        fig_internships = px.line(
            reliable_internships,
            x="internships_count",
            y="placement_rate",
            markers=True,
            custom_data=[
                "sample_size",
                "placed_students",
            ],
        )

        fig_internships.update_traces(
            hovertemplate=(
                "Internships: %{x}<br>"
                "Placement Rate: %{y:.2f}%<br>"
                "Sample Size: %{customdata[0]:,}<br>"
                "Placed Students: %{customdata[1]:,}"
                "<extra></extra>"
            )
        )

        fig_internships.update_layout(
            xaxis_title="Number of Internships",
            yaxis_title="Placement Rate (%)",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_internships,
            use_container_width=True,
        )

        st.caption(
            "Only groups with at least 100 students are shown. "
            "The upward pattern is an association in this synthetic "
            "dataset and should not be interpreted as proof that "
            "internships cause placement."
        )

        unreliable_internships = (
            internship_analysis[
                ~internship_analysis[
                    "reliable_group"
                ]
            ]
        )

        if not unreliable_internships.empty:

            with st.expander(
                "View low-sample internship groups"
            ):

                st.dataframe(
                    unreliable_internships[
                        [
                            "internships_count",
                            "sample_size",
                            "placed_students",
                            "placement_rate",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


    with trend_right:

        st.subheader(
            "Projects vs Placement Rate"
        )

        project_analysis = (
            get_project_analysis(
                df
            )
        )

        reliable_projects = (
            project_analysis[
                project_analysis[
                    "reliable_group"
                ]
            ]
            .copy()
        )

        fig_projects = px.line(
            reliable_projects,
            x="projects_count",
            y="placement_rate",
            markers=True,
            custom_data=[
                "sample_size",
                "placed_students",
            ],
        )

        fig_projects.update_traces(
            hovertemplate=(
                "Projects: %{x}<br>"
                "Placement Rate: %{y:.2f}%<br>"
                "Sample Size: %{customdata[0]:,}<br>"
                "Placed Students: %{customdata[1]:,}"
                "<extra></extra>"
            )
        )

        fig_projects.update_layout(
            xaxis_title="Number of Projects",
            yaxis_title="Placement Rate (%)",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_projects,
            use_container_width=True,
        )

        st.caption(
            "Only groups with at least 100 students are shown. "
            "Higher project counts are associated with higher placement "
            "rates in this synthetic dataset, but this does not establish "
            "a causal relationship."
        )

        unreliable_projects = (
            project_analysis[
                ~project_analysis[
                    "reliable_group"
                ]
            ]
        )

        if not unreliable_projects.empty:

            with st.expander(
                "View low-sample project groups"
            ):

                st.dataframe(
                    unreliable_projects[
                        [
                            "projects_count",
                            "sample_size",
                            "placed_students",
                            "placement_rate",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


    # --------------------------------------------------
    # SALARY ANALYTICS
    # --------------------------------------------------

    st.header(
        "Salary Analytics"
    )

    salary_analysis = (
        get_salary_analysis(df)
    )

    salary1, salary2, salary3, salary4 = (
        st.columns(4)
    )

    salary1.metric(
        "Average Salary",
        f"{salary_analysis['average']:.2f} LPA",
    )

    salary2.metric(
        "Median Salary",
        f"{salary_analysis['median']:.2f} LPA",
    )

    salary3.metric(
        "Highest Salary",
        f"{salary_analysis['maximum']:.2f} LPA",
    )

    salary4.metric(
        "75th Percentile",
        f"{salary_analysis['q3']:.2f} LPA",
    )


    placed_df = df[
        df["placement_status"] == "Placed"
    ]

    st.subheader(
        "Salary Distribution"
    )

    fig_salary = px.histogram(
        placed_df,
        x="salary_package_lpa",
        nbins=30,
    )

    fig_salary.update_layout(
        xaxis_title="Salary Package (LPA)",
        yaxis_title="Number of Students",
        bargap=0.05,
    )

    st.plotly_chart(
        fig_salary,
        use_container_width=True,
    )


# ==================================================
# STUDENT INTELLIGENCE PAGE
# ==================================================

else:

    st.header(
        "Individual Student Intelligence"
    )


    # --------------------------------------------------
    # STUDENT SELECTION
    # --------------------------------------------------

    min_student_id = int(
        df["student_id"].min()
    )

    max_student_id = int(
        df["student_id"].max()
    )

    selected_student_id = st.number_input(
        "Enter Student ID",
        min_value=min_student_id,
        max_value=max_student_id,
        value=min_student_id,
        step=1,
    )


    matching_student = df[
        df["student_id"]
        == selected_student_id
    ]


    if matching_student.empty:

        st.error(
            "Student ID not found."
        )

        st.stop()


    student = (
        matching_student.iloc[0]
    )


    # --------------------------------------------------
    # STUDENT PROFILE
    # --------------------------------------------------

    st.subheader(
        "Student Profile"
    )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "CGPA",
        f"{student['cgpa']:.2f}",
    )

    c2.metric(
        "Coding",
        int(
            student[
                "coding_skill_score"
            ]
        ),
    )

    c3.metric(
        "Projects",
        int(
            student[
                "projects_count"
            ]
        ),
    )

    c4.metric(
        "Internships",
        int(
            student[
                "internships_count"
            ]
        ),
    )

    c5.metric(
        "Current Status",
        student[
            "placement_status"
        ],
    )


    profile_left, profile_right = (
        st.columns(2)
    )

    with profile_left:

        st.write(
            f"**Branch:** "
            f"{student['branch']}"
        )

        st.write(
            f"**College Tier:** "
            f"{student['college_tier']}"
        )

    with profile_right:

        st.write(
            f"**Certifications:** "
            f"{int(student['certifications_count'])}"
        )

        st.write(
            f"**GitHub Repositories:** "
            f"{int(student['github_repos'])}"
        )


    # --------------------------------------------------
    # BENCHMARK ANALYSIS
    # --------------------------------------------------

    benchmarks = (
        get_placed_student_benchmarks(
            df
        )
    )

    analysis = analyze_student(
        student,
        benchmarks,
    )

    recommendations = (
        generate_recommendations(
            analysis
        )
    )


    # --------------------------------------------------
    # STUDENT VS PLACED BENCHMARK
    # --------------------------------------------------

    st.subheader(
        "Student vs Placed-Student Benchmark"
    )

    comparison_rows = []

    selected_features = [
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "logical_reasoning_score",
        "mock_interview_score",
    ]

    feature_names = {
        "coding_skill_score":
            "Coding",

        "aptitude_score":
            "Aptitude",

        "communication_skill_score":
            "Communication",

        "logical_reasoning_score":
            "Logical Reasoning",

        "mock_interview_score":
            "Mock Interview",
    }


    for feature in selected_features:

        comparison_rows.append(
            {
                "Skill":
                    feature_names[
                        feature
                    ],

                "Student":
                    student[
                        feature
                    ],

                "Placed Average":
                    benchmarks[
                        feature
                    ]["mean"],
            }
        )


    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
    )


    fig_comparison = (
        go.Figure()
    )

    fig_comparison.add_trace(
        go.Bar(
            name="Student",
            x=comparison_df[
                "Skill"
            ],
            y=comparison_df[
                "Student"
            ],
        )
    )

    fig_comparison.add_trace(
        go.Bar(
            name="Placed Average",
            x=comparison_df[
                "Skill"
            ],
            y=comparison_df[
                "Placed Average"
            ],
        )
    )

    fig_comparison.update_layout(
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True,
    )


    # --------------------------------------------------
    # STRENGTHS + IMPROVEMENT AREAS
    # --------------------------------------------------

    strength_col, weakness_col = (
        st.columns(2)
    )


    with strength_col:

        st.subheader(
            "Top Strengths"
        )

        if analysis[
            "strengths"
        ]:

            for item in (
                analysis[
                    "strengths"
                ][:5]
            ):

                st.success(
                    f"{item['feature']} "
                    f"— Strength Score: "
                    f"{item['score']:.2f}"
                )

        else:

            st.info(
                "No major strengths identified "
                "relative to the benchmark."
            )


    with weakness_col:

        st.subheader(
            "Priority Improvement Areas"
        )

        if analysis[
            "improvement_areas"
        ]:

            for item in (
                analysis[
                    "improvement_areas"
                ][:5]
            ):

                st.warning(
                    f"{item['feature']} "
                    f"— Priority Score: "
                    f"{item['score']:.2f}"
                )

        else:

            st.info(
                "No major improvement areas "
                "identified relative to the benchmark."
            )


    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------

    st.subheader(
        "Recommended Preparation Actions"
    )

    if recommendations:

        for recommendation in (
            recommendations
        ):

            st.write(
                f"• {recommendation}"
            )

    else:

        st.write(
            "Maintain current performance and "
            "continue strengthening your profile."
        )


    # --------------------------------------------------
    # MACHINE LEARNING ESTIMATE
    # --------------------------------------------------

    st.header(
        "Machine Learning Placement Estimate"
    )


    if model is None:

        st.error(
            "Saved ML model not found. "
            "Run src/main.py first."
        )


    else:

        prediction = (
            predict_student_probability(
                model,
                student,
            )
        )


        p1, p2 = (
            st.columns(2)
        )

        p1.metric(
            "Model Prediction",
            prediction[
                "prediction"
            ],
        )

        p2.metric(
            "Placement Estimate",
            f"{prediction['placement_probability']:.2f}%",
        )


        progress_value = (
            prediction[
                "placement_probability"
            ] / 100
        )

        progress_value = min(
            max(
                progress_value,
                0.0,
            ),
            1.0,
        )

        st.progress(
            progress_value
        )


        st.caption(
            "Prototype estimate based on synthetic placement data. "
            "This should not be treated as a guaranteed "
            "real-world placement probability."
        )


        # ==================================================
        # AI REPORT GENERATION
        # ==================================================

        st.header(
            "✨ AI Placement Readiness Report"
        )

        st.write(
            "Generate a personalized placement-readiness report "
            "using the student's verified analytics, benchmark "
            "comparisons, recommendations and ML estimate."
        )


        if st.button(
            "Generate AI Report",
            type="primary",
        ):

            with st.spinner(
                "Generating placement readiness report..."
            ):

                report_result = (
                    generate_student_report(
                        student,
                        analysis,
                        recommendations,
                        prediction,
                    )
                )


            if (
                report_result[
                    "source"
                ]
                == "groq"
            ):

                st.success(
                    "AI report generated successfully."
                )

            else:

                st.warning(
                    "AI service was unavailable, so the "
                    "built-in fallback report was generated."
                )


            st.markdown(
                report_result[
                    "report"
                ]
            )


            with st.expander(
                "Report generation details"
            ):

                st.write(
                    f"Source: "
                    f"{report_result['source']}"
                )

                if (
                    report_result[
                        "source"
                    ]
                    != "groq"
                ):

                    st.write(
                        report_result[
                            "message"
                        ]
                    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Placement Intelligence System | "
    "Prototype using synthetic placement data."
)