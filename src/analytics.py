MIN_GROUP_SIZE = 100


def get_overall_metrics(df):
    total_students = len(df)

    placed_students = (
        df["placement_status"] == "Placed"
    ).sum()

    not_placed_students = (
        df["placement_status"] == "Not Placed"
    ).sum()

    placement_rate = (
        placed_students / total_students
    ) * 100

    placed_salary = df.loc[
        df["placement_status"] == "Placed",
        "salary_package_lpa"
    ]

    average_salary = placed_salary.mean()
    highest_salary = placed_salary.max()

    return {
        "total_students": total_students,
        "placed_students": placed_students,
        "not_placed_students": not_placed_students,
        "placement_rate": placement_rate,
        "average_salary": average_salary,
        "highest_salary": highest_salary,
    }


def get_skill_comparison(df):
    skill_columns = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "logical_reasoning_score",
        "mock_interview_score",
        "projects_count",
        "internships_count",
    ]

    comparison = (
        df.groupby(
            "placement_status"
        )[skill_columns]
        .mean()
    )

    return comparison


def get_branch_analysis(df):
    branch_analysis = (
        df.groupby("branch")["placement_status"]
        .apply(
            lambda x: (
                x == "Placed"
            ).mean() * 100
        )
        .sort_values(
            ascending=False
        )
    )

    return branch_analysis


def get_college_tier_analysis(df):
    tier_analysis = (
        df.groupby(
            "college_tier"
        )["placement_status"]
        .apply(
            lambda x: (
                x == "Placed"
            ).mean() * 100
        )
        .sort_values(
            ascending=False
        )
    )

    return tier_analysis


def get_skill_gap_analysis(df):
    skill_columns = [
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
        "hackathons_participated",
    ]

    placed = df[
        df["placement_status"] == "Placed"
    ]

    not_placed = df[
        df["placement_status"] == "Not Placed"
    ]

    gaps = {}

    for column in skill_columns:
        placed_average = (
            placed[column].mean()
        )

        not_placed_average = (
            not_placed[column].mean()
        )

        gaps[column] = {
            "placed_average":
                placed_average,

            "not_placed_average":
                not_placed_average,

            "gap":
                placed_average
                - not_placed_average,
        }

    return gaps


def get_salary_analysis(df):
    placed_students = df[
        df["placement_status"] == "Placed"
    ]

    salary = (
        placed_students[
            "salary_package_lpa"
        ]
    )

    return {
        "average":
            salary.mean(),

        "median":
            salary.median(),

        "minimum":
            salary.min(),

        "maximum":
            salary.max(),

        "q1":
            salary.quantile(0.25),

        "q3":
            salary.quantile(0.75),
    }


def get_work_experience_analysis(
    df,
    min_group_size=MIN_GROUP_SIZE,
):
    grouped = (
        df.groupby(
            "internships_count"
        )
        .agg(
            sample_size=(
                "placement_status",
                "size",
            ),

            placed_students=(
                "placement_status",
                lambda x: (
                    x == "Placed"
                ).sum(),
            ),
        )
        .reset_index()
    )

    grouped[
        "placement_rate"
    ] = (
        grouped[
            "placed_students"
        ]
        / grouped[
            "sample_size"
        ]
    ) * 100

    grouped[
        "reliable_group"
    ] = (
        grouped[
            "sample_size"
        ]
        >= min_group_size
    )

    return grouped.sort_values(
        "internships_count"
    )


def get_project_analysis(
    df,
    min_group_size=MIN_GROUP_SIZE,
):
    grouped = (
        df.groupby(
            "projects_count"
        )
        .agg(
            sample_size=(
                "placement_status",
                "size",
            ),

            placed_students=(
                "placement_status",
                lambda x: (
                    x == "Placed"
                ).sum(),
            ),
        )
        .reset_index()
    )

    grouped[
        "placement_rate"
    ] = (
        grouped[
            "placed_students"
        ]
        / grouped[
            "sample_size"
        ]
    ) * 100

    grouped[
        "reliable_group"
    ] = (
        grouped[
            "sample_size"
        ]
        >= min_group_size
    )

    return grouped.sort_values(
        "projects_count"
    )