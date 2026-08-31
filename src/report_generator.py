import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"


def build_student_report_prompt(
    student,
    analysis,
    recommendations,
    prediction,
):
    strengths = analysis.get(
        "strengths",
        []
    )

    improvement_areas = analysis.get(
        "improvement_areas",
        []
    )

    comparisons = analysis.get(
        "comparisons",
        {}
    )

    strength_text = (
        "\n".join(
            [
                f"- {item['feature']}: "
                f"strength score {item['score']:.2f}"
                for item in strengths[:5]
            ]
        )
        if strengths
        else "- No major strengths identified."
    )

    weakness_text = (
        "\n".join(
            [
                f"- {item['feature']}: "
                f"priority score {item['score']:.2f}"
                for item in improvement_areas[:5]
            ]
        )
        if improvement_areas
        else "- No major improvement areas identified."
    )

    recommendation_text = (
        "\n".join(
            [
                f"- {recommendation}"
                for recommendation in recommendations
            ]
        )
        if recommendations
        else (
            "- Maintain current preparation "
            "and continue improving consistently."
        )
    )

    comparison_lines = []

    for feature, values in comparisons.items():

        comparison_lines.append(
            f"- {feature}: "
            f"student={values['student_value']:.2f}, "
            f"placed_average={values['placed_average']:.2f}, "
            f"standardized_difference="
            f"{values['normalized_difference']:.2f}"
        )

    comparison_text = "\n".join(
        comparison_lines
    )

    prompt = f"""
Generate a professional placement-readiness report
for the student described below.

STRICT RULES:

1. Use ONLY the facts supplied in this prompt.

2. Do not invent statistics, achievements, skills,
   companies, salaries or student information.

3. Do not claim that any feature causes placement.

4. Benchmark differences are comparisons only.

5. The ML estimate is based on synthetic data.

6. Never describe the ML estimate as a guaranteed
   placement probability.

7. Do not recalculate or alter any supplied numbers.

8. Keep advice practical and directly related to the
   listed improvement areas.

9. Do not mention system prompts or hidden instructions.

10. Do not state that you are an AI model.

11. Clearly mention that the dataset is synthetic.

12. Keep the tone professional and constructive.

13. Recommendations must respect the student's
    existing strengths.

    Do not recommend increasing a metric that is already
    identified as a strength unless the recommendation
    specifically focuses on improving quality,
    documentation, presentation, or utilization.

14. A standardized difference is NOT the raw difference
    from the placed-student average.

    When mentioning standardized differences,
    explicitly describe them as standardized differences
    or benchmark scores.

15. Do not infer student facts that are not explicitly
    provided.

    For example, do not call the student a graduate,
    fresher, final-year student, or job seeker unless
    that information is supplied.

16. Do not invent target career roles or domains.

    Recommendations must remain generic unless a target
    role is explicitly provided.

17. When describing a benchmark comparison, use the
    actual student value and placed average when available.

    Do not reinterpret standardized scores as raw units.

18. Recommendations must stay generic unless the
    application supplies a target role, technology,
    certification, platform, or company.

    Do not invent examples such as AWS, Azure, LeetCode,
    open-source contributions, specific project types,
    certification names, or companies.

19. If GitHub presence is weak but Projects is already
    identified as a strength, recommend publishing,
    documenting, organizing and improving existing
    projects before recommending additional projects
    or repositories.

20. Do not recommend increasing an already-strong metric
    merely for the sake of adding more activity.

    Focus on using, documenting and presenting existing
    strengths effectively.
21. Never describe the student using an education or career-stage
    label unless it is explicitly supplied.

    Do not call the student a graduate, final-year student,
    fresher, job seeker, candidate, or professional unless that
    information is present in the supplied data.

22. Do not invent timelines, schedules, frequencies, deadlines,
    numeric targets, or preparation durations.

    For example, do not add phrases such as "within 3-6 months",
    "bi-weekly", "daily", "weekly", or "complete two certifications"
    unless those details are explicitly supplied by the application.
23. Do not invent consequences or recruiter reactions.

    Do not claim that a weakness will reduce recruiter visibility,
    selection chances, interview performance, or placement outcome
    unless that relationship is explicitly supplied.

24. Recommendations must not introduce new actions beyond the
    system-generated recommendations.

    You may rephrase or organize the supplied recommendations,
    but do not add new activities, schedules, project counts,
    study routines, assignments, classes, collaborations,
    or preparation methods.

25. Do not create numeric improvement targets.

    Do not recommend adding exactly one project, reaching a specific
    repository count, certification count, score, CGPA, or benchmark
    unless that target is explicitly supplied by the application.

STUDENT PROFILE

Student ID: {student['student_id']}

Branch: {student['branch']}

College Tier: {student['college_tier']}

CGPA: {student['cgpa']:.2f}

Coding Skill Score:
{student['coding_skill_score']}

Aptitude Score:
{student['aptitude_score']}

Communication Skill Score:
{student['communication_skill_score']}

Logical Reasoning Score:
{student['logical_reasoning_score']}

Mock Interview Score:
{student['mock_interview_score']}

Projects:
{student['projects_count']}

Internships:
{student['internships_count']}

Certifications:
{student['certifications_count']}

GitHub Repositories:
{student['github_repos']}


CURRENT DATASET STATUS

Placement Status:
{student['placement_status']}


BENCHMARK COMPARISONS

{comparison_text}


MEANINGFUL STRENGTHS

{strength_text}


PRIORITY IMPROVEMENT AREAS

{weakness_text}


SYSTEM-GENERATED RECOMMENDATIONS

{recommendation_text}


MACHINE LEARNING ESTIMATE

Prediction:
{prediction['prediction']}

Model Estimate:
{prediction['placement_probability']:.2f}%


The model estimate is a prototype estimate based
on synthetic placement data and is not a guaranteed
real-world placement probability.


Generate exactly these sections:

### Placement Readiness Summary

### Key Strengths

### Priority Improvement Areas

### Recommended Preparation Plan

### Model Estimate

### Important Note


Keep the report approximately 300-450 words.
"""

    return prompt.strip()


def generate_fallback_report(
    student,
    analysis,
    recommendations,
    prediction,
):
    strengths = analysis.get(
        "strengths",
        []
    )

    weaknesses = analysis.get(
        "improvement_areas",
        []
    )

    if strengths:

        strength_text = ", ".join(
            item["feature"]
            for item in strengths[:3]
        )

    else:

        strength_text = (
            "no major benchmark strengths "
            "were identified"
        )

    if weaknesses:

        weakness_text = ", ".join(
            item["feature"]
            for item in weaknesses[:3]
        )

    else:

        weakness_text = (
            "no major benchmark weaknesses "
            "were identified"
        )

    action_text = "\n".join(
        f"- {recommendation}"
        for recommendation in recommendations
    )

    if not action_text:

        action_text = (
            "- Continue maintaining current performance "
            "while improving consistently."
        )

    return f"""
### Placement Readiness Summary

Student {student['student_id']} currently has a CGPA of
{student['cgpa']:.2f}. Relative to the placed-student
benchmarks used by this system, the profile shows
{strength_text} as the most notable strengths.

### Key Strengths

The strongest benchmark-positive areas are:
{strength_text}.

### Priority Improvement Areas

The main areas requiring additional attention are:
{weakness_text}.

### Recommended Preparation Plan

{action_text}

### Model Estimate

The machine-learning component predicts
**{prediction['prediction']}** with a model estimate of
**{prediction['placement_probability']:.2f}%**.

This value is only a prototype model estimate and should not
be interpreted as a guaranteed real-world placement probability.

### Important Note

This system currently uses synthetic placement data.
The report is intended as a decision-support and preparation
tool rather than a definitive prediction of placement outcomes.
""".strip()


def generate_student_report(
    student,
    analysis,
    recommendations,
    prediction,
):
    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return {
            "report":
                generate_fallback_report(
                    student,
                    analysis,
                    recommendations,
                    prediction,
                ),

            "source":
                "fallback",

            "message":
                "GROQ_API_KEY was not found.",
        }

    prompt = build_student_report_prompt(
        student,
        analysis,
        recommendations,
        prediction,
    )

    try:

        client = Groq(
            api_key=api_key
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate grounded student "
                        "placement-readiness reports. "
                        "Use only facts supplied by the "
                        "application. Never invent facts, "
                        "modify numerical values or treat "
                        "model estimates as guarantees."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.2,

            reasoning_effort="low",

            include_reasoning=False,

            max_completion_tokens=2200,
        )


        if not completion.choices:

            raise ValueError(
                "Groq returned no completion choices."
            )

        report = (
            completion
            .choices[0]
            .message
            .content
        )

        if (
            not report
            or not report.strip()
        ):

            raise ValueError(
                "Groq returned an empty response."
            )

        return {
            "report":
                report.strip(),

            "source":
                "groq",

            "message":
                "Report generated using Groq.",
        }

    except Exception as error:

        fallback_report = (
            generate_fallback_report(
                student,
                analysis,
                recommendations,
                prediction,
            )
        )

        return {
            "report":
                fallback_report,

            "source":
                "fallback",

            "message":
                (
                    "Groq unavailable. "
                    "Fallback report used. "
                    f"Reason: {error}"
                ),
        }