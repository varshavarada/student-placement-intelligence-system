from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_placement_distribution(df):
    counts = df["placement_status"].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")

    plt.title("Placement Status Distribution")
    plt.xlabel("Placement Status")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "placement_distribution.png"
    )

    plt.close()


def plot_branch_placement_rate(branch_analysis):
    plt.figure(figsize=(9, 5))

    branch_analysis.sort_values().plot(
        kind="barh"
    )

    plt.title("Branch-wise Placement Rate")
    plt.xlabel("Placement Rate (%)")
    plt.ylabel("Branch")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "branch_placement_rate.png"
    )

    plt.close()


def plot_salary_distribution(df):
    salary = df.loc[
        df["placement_status"] == "Placed",
        "salary_package_lpa"
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(
        salary,
        bins=25
    )

    plt.title("Salary Package Distribution")
    plt.xlabel("Salary Package (LPA)")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "salary_distribution.png"
    )

    plt.close()