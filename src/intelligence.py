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


BENCHMARK_COLUMNS = list(
    FEATURE_LABELS.keys()
)


# --------------------------------------------------
# THRESHOLDS
# --------------------------------------------------

STRENGTH_THRESHOLD = 0.25
IMPROVEMENT_THRESHOLD = -0.25


# --------------------------------------------------
# PLACED STUDENT BENCHMARKS
# --------------------------------------------------

def get_placed_student_benchmarks(df):

    placed_students = df[
        df["placement_status"] == "Placed"
    ]

    benchmarks = {}

    for column in BENCHMARK_COLUMNS:

        benchmarks[column] = {
            "mean":
                placed_students[column].mean(),

            "std":
                placed_students[column].std(),
        }

    return benchmarks


# --------------------------------------------------
# STUDENT ANALYSIS
# --------------------------------------------------

def analyze_student(
    student,
    benchmarks,
):

    analysis = {
        "strengths": [],
        "improvement_areas": [],
        "comparisons": {},
    }

    for (
        feature,
        benchmark_data,
    ) in benchmarks.items():

        student_value = (
            student[feature]
        )

        benchmark_mean = (
            benchmark_data["mean"]
        )

        benchmark_std = (
            benchmark_data["std"]
        )


        raw_difference = (
            student_value
            - benchmark_mean
        )


        if benchmark_std > 0:

            normalized_difference = (
                raw_difference
                / benchmark_std
            )

        else:

            normalized_difference = 0


        # Store detailed comparison
        analysis["comparisons"][feature] = {
            "student_value":
                student_value,

            "placed_average":
                benchmark_mean,

            "raw_difference":
                raw_difference,

            "normalized_difference":
                normalized_difference,
        }


        item = {
            "feature":
                FEATURE_LABELS[feature],

            "raw_difference":
                raw_difference,

            "score":
                abs(
                    normalized_difference
                ),
        }


        # Meaningful strength only
        if (
            normalized_difference
            >= STRENGTH_THRESHOLD
        ):

            analysis[
                "strengths"
            ].append(
                item
            )


        # Meaningful improvement area only
        elif (
            normalized_difference
            <= IMPROVEMENT_THRESHOLD
        ):

            analysis[
                "improvement_areas"
            ].append(
                item
            )


    # Strongest strengths first
    analysis[
        "strengths"
    ].sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    # Highest-priority weaknesses first
    analysis[
        "improvement_areas"
    ].sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    return analysis


# --------------------------------------------------
# RECOMMENDATION ENGINE
# --------------------------------------------------

def generate_recommendations(
    analysis,
):

    recommendations = []

    weak_areas = (
        analysis[
            "improvement_areas"
        ]
    )


    recommendation_map = {

        "Coding Skill":
            "Improve coding ability through regular "
            "problem solving and practical development.",


        "Aptitude":
            "Prioritize quantitative aptitude and "
            "placement-test practice.",


        "Communication":
            "Practice structured speaking, group "
            "discussions and interview communication.",


        "Logical Reasoning":
            "Strengthen logical reasoning through "
            "timed analytical problem-solving practice.",


        "Mock Interview":
            "Increase mock interview practice and "
            "review performance after every session.",


        "Projects":
            "Build additional practical projects "
            "and document them clearly in your portfolio.",


        "Internships":
            "Gain practical exposure through internships "
            "or industry-oriented projects.",


        "GitHub Repositories":
            "Publish and improve your strongest existing projects on GitHub "
            "with clear README files, documentation and regular version control.",


        "Certifications":
            "Complete a relevant certification that supports your current "
            "technical profile and placement preparation.",


        "CGPA":
            "Improve academic performance where possible "
            "while maintaining placement preparation.",
    }


    # Maximum 5 priority recommendations
    for item in weak_areas[:5]:

        feature = (
            item["feature"]
        )

        if (
            feature
            in recommendation_map
        ):

            recommendations.append(
                recommendation_map[
                    feature
                ]
            )


    return recommendations