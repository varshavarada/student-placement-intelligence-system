MIN_GROUP_SIZE = 100


def get_overall_metrics(df):
    total_students = len(df)

    placed_students = int(
        (
            df["placement_status"]
            == "Placed"
        ).sum()
    )

    not_placed_students = int(
        (
            df["placement_status"]
            == "Not Placed"
        ).sum()
    )

    placement_rate = (
        placed_students
        / total_students
        * 100
        if total_students > 0
        else 0
    )

    average_salary = None
    highest_salary = None

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

            highest_salary = (
                placed_salary.max()
            )

    return {
        "total_students":
            total_students,

        "placed_students":
            placed_students,

        "not_placed_students":
            not_placed_students,

        "placement_rate":
            placement_rate,

        "average_salary":
            average_salary,

        "highest_salary":
            highest_salary,
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

    available_columns = [
        column
        for column in skill_columns
        if column in df.columns
    ]

    if not available_columns:
        return None

    comparison = (
        df.groupby(
            "placement_status"
        )[available_columns]
        .mean()
    )

    return comparison


def get_branch_analysis(df):
    if "branch" not in df.columns:
        return None

    branch_analysis = (
        df.groupby(
            "branch"
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

    branch_analysis[
        "placement_rate"
    ] = (
        branch_analysis[
            "placed_students"
        ]
        / branch_analysis[
            "sample_size"
        ]
    ) * 100

    return (
        branch_analysis
        .sort_values(
            "placement_rate",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def get_college_tier_analysis(df):
    if "college_tier" not in df.columns:
        return None

    tier_analysis = (
        df.groupby(
            "college_tier"
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

    tier_analysis[
        "placement_rate"
    ] = (
        tier_analysis[
            "placed_students"
        ]
        / tier_analysis[
            "sample_size"
        ]
    ) * 100

    return (
        tier_analysis
        .sort_values(
            "placement_rate",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


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

    available_columns = [
        column
        for column in skill_columns
        if column in df.columns
    ]

    placed = df[
        df["placement_status"]
        == "Placed"
    ]

    not_placed = df[
        df["placement_status"]
        == "Not Placed"
    ]

    gaps = {}

    for column in available_columns:

        placed_average = (
            placed[column]
            .mean()
        )

        not_placed_average = (
            not_placed[column]
            .mean()
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
    if (
        "salary_package_lpa"
        not in df.columns
    ):
        return None

    placed_students = df[
        df["placement_status"]
        == "Placed"
    ]

    salary = (
        placed_students[
            "salary_package_lpa"
        ]
        .dropna()
    )

    if salary.empty:
        return None

    return {
        "average_salary":
            salary.mean(),

        "median_salary":
            salary.median(),

        "minimum_salary":
            salary.min(),

        "highest_salary":
            salary.max(),

        "q1_salary":
            salary.quantile(
                0.25
            ),

        "q3_salary":
            salary.quantile(
                0.75
            ),
    }


def get_work_experience_analysis(
    df,
    min_group_size=MIN_GROUP_SIZE,
):

    if (
        "internships_count"
        not in df.columns
    ):
        return None

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

    return (
        grouped
        .sort_values(
            "internships_count"
        )
        .reset_index(
            drop=True
        )
    )


def get_project_analysis(
    df,
    min_group_size=MIN_GROUP_SIZE,
):

    if (
        "projects_count"
        not in df.columns
    ):
        return None

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

    return (
        grouped
        .sort_values(
            "projects_count"
        )
        .reset_index(
            drop=True
        )
    )