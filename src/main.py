from data_loader import load_data

from analytics import (
    get_overall_metrics,
    get_skill_comparison,
    get_branch_analysis,
    get_college_tier_analysis,
    get_skill_gap_analysis,
    get_salary_analysis,
    get_work_experience_analysis,
    get_project_analysis,
)

from visualization import (
    plot_placement_distribution,
    plot_branch_placement_rate,
    plot_salary_distribution,
)

from intelligence import (
    get_placed_student_benchmarks,
    analyze_student,
    generate_recommendations,
)

from ml_model import (
    get_or_train_model,
    predict_student_probability,
)

def main():

    print(
        "=== STUDENT PLACEMENT "
        "INTELLIGENCE SYSTEM ==="
    )

    df = load_data()

    # --------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------

    metrics = get_overall_metrics(df)

    print("\n=== OVERALL PLACEMENT METRICS ===")

    print(
        f"Total Students: "
        f"{metrics['total_students']}"
    )

    print(
        f"Placed Students: "
        f"{metrics['placed_students']}"
    )

    print(
        f"Not Placed Students: "
        f"{metrics['not_placed_students']}"
    )

    print(
        f"Placement Rate: "
        f"{metrics['placement_rate']:.2f}%"
    )

    print(
        f"Average Salary Package: "
        f"{metrics['average_salary']:.2f} LPA"
    )

    print(
        f"Highest Salary Package: "
        f"{metrics['highest_salary']:.2f} LPA"
    )

    # --------------------------------------------------
    # PLACED VS NOT PLACED COMPARISON
    # --------------------------------------------------

    print(
        "\n=== PLACED VS NOT PLACED COMPARISON ==="
    )

    skill_comparison = get_skill_comparison(df)

    print(
        skill_comparison.round(2).to_string()
    )

    # --------------------------------------------------
    # BRANCH ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== BRANCH-WISE PLACEMENT RATE ==="
    )

    branch_analysis = get_branch_analysis(df)

    print(
        branch_analysis.round(2).to_string()
    )

    # --------------------------------------------------
    # COLLEGE TIER ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== COLLEGE TIER PLACEMENT RATE ==="
    )

    tier_analysis = get_college_tier_analysis(df)

    print(
        tier_analysis.round(2).to_string()
    )

    # --------------------------------------------------
    # SKILL GAP ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== SKILL GAP ANALYSIS ==="
    )

    skill_gaps = get_skill_gap_analysis(df)

    sorted_gaps = sorted(
        skill_gaps.items(),
        key=lambda item: abs(item[1]["gap"]),
        reverse=True,
    )

    for skill, values in sorted_gaps:

        print(
            f"{skill}: "
            f"Placed={values['placed_average']:.2f}, "
            f"Not Placed="
            f"{values['not_placed_average']:.2f}, "
            f"Gap={values['gap']:.2f}"
        )

    # --------------------------------------------------
    # SALARY ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== SALARY ANALYSIS ==="
    )

    salary = get_salary_analysis(df)

    print(
        f"Average Salary: "
        f"{salary['average']:.2f} LPA"
    )

    print(
        f"Median Salary: "
        f"{salary['median']:.2f} LPA"
    )

    print(
        f"Minimum Salary: "
        f"{salary['minimum']:.2f} LPA"
    )

    print(
        f"Maximum Salary: "
        f"{salary['maximum']:.2f} LPA"
    )

    print(
        f"25th Percentile: "
        f"{salary['q1']:.2f} LPA"
    )

    print(
        f"75th Percentile: "
        f"{salary['q3']:.2f} LPA"
    )

    # --------------------------------------------------
    # INTERNSHIP ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== INTERNSHIPS VS PLACEMENT ==="
    )

    internship_analysis = (
        get_work_experience_analysis(df)
    )

    print(
        internship_analysis
        .round(2)
        .to_string()
    )

    # --------------------------------------------------
    # PROJECT ANALYSIS
    # --------------------------------------------------

    print(
        "\n=== PROJECTS VS PLACEMENT ==="
    )

    project_analysis = (
        get_project_analysis(df)
    )

    print(
        project_analysis
        .round(2)
        .to_string()
    )

    # --------------------------------------------------
    # INDIVIDUAL STUDENT INTELLIGENCE
    # --------------------------------------------------

    print(
        "\n=== INDIVIDUAL STUDENT INTELLIGENCE ==="
    )

    benchmarks = get_placed_student_benchmarks(df)

    # Temporary test student
    student = df.iloc[0]

    analysis = analyze_student(
        student,
        benchmarks,
    )

    recommendations = generate_recommendations(
        analysis
    )

    print(
        f"\nStudent ID: "
        f"{student['student_id']}"
    )

    print(
        f"Current Status: "
        f"{student['placement_status']}"
    )

    print("\nTop Strengths:")

    if analysis["strengths"]:

        for item in analysis["strengths"][:5]:

            print(
                f"- {item['feature']} "
                f"(strength score: "
                f"{item['score']:.2f})"
            )

    else:
        print(
            "- No major strengths identified "
            "relative to the placed-student benchmark."
        )

    print(
        "\nPriority Improvement Areas:"
    )

    if analysis["improvement_areas"]:

        for item in analysis[
            "improvement_areas"
        ][:5]:

            print(
                f"- {item['feature']} "
                f"(priority score: "
                f"{item['score']:.2f})"
            )

    else:
        print(
            "- No major improvement areas identified "
            "relative to the placed-student benchmark."
        )

    print("\nRecommended Actions:")

    if recommendations:

        for recommendation in recommendations:

            print(
                f"- {recommendation}"
            )

    else:
        print(
            "- Maintain current performance and "
            "continue strengthening your portfolio."
        )
    # --------------------------------------------------
    # MACHINE LEARNING MODEL COMPARISON
    # --------------------------------------------------

    print(
        "\n=== MACHINE LEARNING PLACEMENT MODEL ==="
    )

    ml_result = get_or_train_model(df)

    model = ml_result["pipeline"]

    if ml_result["source"] == "trained":

        print(
            "\nModel trained and saved successfully."
        )

        comparison = ml_result[
            "comparison"
        ]

        print("\nModel Comparison:")

        for model_name, metrics in (
                comparison["results"].items()
        ):
            print(f"\n{model_name}")

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"F1 Score: "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"ROC-AUC: "
                f"{metrics['roc_auc']:.4f}"
            )

        print(
            f"\nSelected Model: "
            f"{comparison['best_model_name']}"
        )

    else:

        print(
            "\nSaved model loaded successfully."
        )

    student_prediction = (
        predict_student_probability(
            model,
            student,
        )
    )

    print(
        "\nStudent Placement Prediction:"
    )

    print(
        f"Predicted Status: "
        f"{student_prediction['prediction']}"
    )

    print(
        f"Placement Probability: "
        f"{student_prediction['placement_probability']:.2f}%"
    )
    # --------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------

    print(
        "\nGenerating visualizations..."
    )

    plot_placement_distribution(df)

    plot_branch_placement_rate(
        branch_analysis
    )

    plot_salary_distribution(df)

    print(
        "Visualizations generated successfully."
    )

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print(
        "\n=== SYSTEM EXECUTION COMPLETED ==="
    )


if __name__ == "__main__":
    main()