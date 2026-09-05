import re


# ==================================================
# SUPPORTED SKILLS
# ==================================================

SKILL_ALIASES = {
    "python": [
        "python",
    ],

    "java": [
        "java",
    ],

    "javascript": [
        "javascript",
        "js",
    ],

    "html": [
        "html",
        "html5",
    ],

    "css": [
        "css",
        "css3",
    ],

    "sql": [
        "sql",
        "mysql",
        "postgresql",
        "postgres",
    ],

    "mongodb": [
        "mongodb",
        "mongo db",
    ],

    "react": [
        "react",
        "react.js",
        "reactjs",
    ],

    "node.js": [
        "node.js",
        "nodejs",
        "node js",
    ],

    "flask": [
        "flask",
    ],

    "django": [
        "django",
    ],

    "streamlit": [
        "streamlit",
    ],

    "pandas": [
        "pandas",
    ],

    "numpy": [
        "numpy",
    ],

    "matplotlib": [
        "matplotlib",
    ],

    "plotly": [
        "plotly",
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn",
        "scikit learn",
    ],

    "machine learning": [
        "machine learning",
        "ml",
    ],

    "deep learning": [
        "deep learning",
        "dl",
    ],

    "data analysis": [
        "data analysis",
        "data analytics",
    ],

    "data visualization": [
        "data visualization",
        "data visualisation",
    ],

    "statistics": [
        "statistics",
        "statistical analysis",
    ],

    "natural language processing": [
        "natural language processing",
        "nlp",
    ],

    "large language models": [
        "large language model",
        "large language models",
        "llm",
        "llms",
    ],

    "retrieval augmented generation": [
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "rag",
    ],

    "git": [
        "git",
    ],

    "github": [
        "github",
    ],

    "docker": [
        "docker",
    ],

    "aws": [
        "aws",
        "amazon web services",
    ],

    "azure": [
        "azure",
        "microsoft azure",
    ],

    "power bi": [
        "power bi",
        "powerbi",
    ],

    "tableau": [
        "tableau",
    ],

    "excel": [
        "excel",
        "microsoft excel",
        "ms excel",
    ],

    "communication": [
        "communication",
        "communication skills",
    ],

    "leadership": [
        "leadership",
        "team leadership",
    ],

    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
}


# ==================================================
# CATEGORY KEYWORDS
# ==================================================

REQUIRED_KEYWORDS = [
    "required",
    "required skill",
    "required skills",
    "must have",
    "must-have",
    "mandatory",
    "essential",
    "minimum requirement",
    "minimum requirements",
    "minimum qualification",
    "minimum qualifications",
]


PREFERRED_KEYWORDS = [
    "preferred",
    "preferred skill",
    "preferred skills",
    "nice to have",
    "nice-to-have",
    "good to have",
    "good-to-have",
    "desirable",
    "plus",
    "advantage",
]


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(
        text
    ).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ==================================================
# LINE NORMALIZATION
# ==================================================

def normalize_line(text):

    if text is None:
        return ""

    text = str(
        text
    ).lower()

    text = text.replace(
        "\t",
        " ",
    )

    text = re.sub(
        r"[ ]+",
        " ",
        text,
    )

    return text.strip()


# ==================================================
# TERM MATCHING
# ==================================================

def contains_term(
    text,
    term,
):

    normalized_text = normalize_text(
        text
    )

    normalized_term = normalize_text(
        term
    )

    if not normalized_term:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(
            normalized_term
        )
        + r"(?![a-z0-9])"
    )

    return (
        re.search(
            pattern,
            normalized_text,
        )
        is not None
    )


# ==================================================
# SKILL EXTRACTION
# ==================================================

def extract_skills(text):

    detected_skills = []

    for (
        canonical_skill,
        aliases,
    ) in SKILL_ALIASES.items():

        found = False

        for alias in aliases:

            if contains_term(
                text,
                alias,
            ):

                found = True
                break

        if found:

            detected_skills.append(
                canonical_skill
            )

    return sorted(
        set(
            detected_skills
        )
    )


# ==================================================
# RESUME ANALYSIS
# ==================================================

def analyze_resume_text(
    resume_text,
):

    normalized_text = normalize_text(
        resume_text
    )

    skills = extract_skills(
        normalized_text
    )

    return {
        "skills": skills,

        "skill_count": len(
            skills
        ),

        "has_content": bool(
            normalized_text
        ),
    }


# ==================================================
# CATEGORY DETECTION
# ==================================================

def detect_explicit_category(text):

    normalized_text = normalize_text(
        text
    )

    if not normalized_text:
        return None

    if any(
        keyword in normalized_text
        for keyword in REQUIRED_KEYWORDS
    ):
        return "required"

    if any(
        keyword in normalized_text
        for keyword in PREFERRED_KEYWORDS
    ):
        return "preferred"

    return None


# ==================================================
# SECTION HEADING DETECTION
# ==================================================

def detect_section_heading(line):

    normalized_line = normalize_line(
        line
    )

    if not normalized_line:
        return None

    cleaned_heading = re.sub(
        r"[:\-–—]+$",
        "",
        normalized_line,
    ).strip()

    if len(
        cleaned_heading.split()
    ) > 8:
        return None

    if any(
        keyword == cleaned_heading
        or cleaned_heading.startswith(
            keyword + " "
        )
        for keyword in REQUIRED_KEYWORDS
    ):
        return "required"

    if any(
        keyword == cleaned_heading
        or cleaned_heading.startswith(
            keyword + " "
        )
        for keyword in PREFERRED_KEYWORDS
    ):
        return "preferred"

    neutral_headings = {
        "skills",
        "technical skills",
        "responsibilities",
        "responsibility",
        "job responsibilities",
        "role responsibilities",
        "qualifications",
        "qualification",
        "about the role",
        "about you",
        "experience",
        "additional experience",
        "additional skills",
        "other skills",
        "other experience",
        "education",
        "job description",
        "role description",
        "what you will do",
        "what you'll do",
        "what we offer",
    }

    if cleaned_heading in neutral_headings:
        return "general"

    return None


# ==================================================
# CATEGORY ASSIGNMENT
# ==================================================

def assign_skill_category(
    category_map,
    skill,
    category,
):

    existing_category = (
        category_map.get(
            skill
        )
    )

    # Priority:
    # Required > Preferred > General

    if category == "required":

        category_map[
            skill
        ] = "required"

    elif category == "preferred":

        if (
            existing_category
            != "required"
        ):
            category_map[
                skill
            ] = "preferred"

    else:

        if existing_category is None:
            category_map[
                skill
            ] = "general"


# ==================================================
# JOB DESCRIPTION ANALYSIS
# ==================================================

def analyze_job_description(
    job_description,
):

    raw_text = ""

    if job_description is not None:
        raw_text = str(
            job_description
        )

    normalized_text = normalize_text(
        raw_text
    )

    all_detected_skills = extract_skills(
        raw_text
    )

    if not normalized_text:

        return {
            "all_skills": [],
            "required_skills": [],
            "preferred_skills": [],
            "general_skills": [],
            "skill_count": 0,
            "has_content": False,
        }

    category_map = {}

    # ==================================================
    # PASS 1:
    # LINE + SECTION BASED CLASSIFICATION
    #
    # Example:
    #
    # Required Skills:
    # Python
    # SQL
    #
    # Preferred Skills:
    # Power BI
    # ==================================================

    raw_lines = (
        raw_text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .split(
            "\n"
        )
    )

    current_section = None

    for raw_line in raw_lines:

        line = normalize_line(
            raw_line
        )

        # Blank line ends current section.
        # This prevents unrelated later text
        # from inheriting Required/Preferred.

        if not line:

            current_section = None
            continue

        heading_category = (
            detect_section_heading(
                line
            )
        )

        if (
            heading_category
            == "required"
        ):
            current_section = (
                "required"
            )

        elif (
            heading_category
            == "preferred"
        ):
            current_section = (
                "preferred"
            )

        elif (
            heading_category
            == "general"
        ):
            current_section = None

        explicit_category = (
            detect_explicit_category(
                line
            )
        )

        if explicit_category is not None:

            line_category = (
                explicit_category
            )

        elif current_section is not None:

            line_category = (
                current_section
            )

        else:

            line_category = (
                "general"
            )

        skills_in_line = extract_skills(
            line
        )

        for skill in skills_in_line:

            assign_skill_category(
                category_map,
                skill,
                line_category,
            )

    # ==================================================
    # PASS 2:
    # SENTENCE / CLAUSE BASED EXPLICIT WORDING
    #
    # Example:
    # Python and SQL are required.
    # Power BI is preferred.
    # ==================================================

    segments = re.split(
        r"[.;\n]+",
        raw_text,
    )

    for segment in segments:

        segment = (
            segment.strip()
        )

        if not segment:
            continue

        explicit_category = (
            detect_explicit_category(
                segment
            )
        )

        # Only explicit category statements
        # are processed in this pass.

        if explicit_category is None:
            continue

        skills_in_segment = (
            extract_skills(
                segment
            )
        )

        for skill in skills_in_segment:

            assign_skill_category(
                category_map,
                skill,
                explicit_category,
            )

    # ==================================================
    # PASS 3:
    # UNCLASSIFIED SKILLS = GENERAL
    # ==================================================

    for skill in all_detected_skills:

        if skill not in category_map:

            category_map[
                skill
            ] = "general"

    # ==================================================
    # BUILD FINAL CATEGORY LISTS
    # ==================================================

    required_skills = sorted(
        skill
        for skill, category
        in category_map.items()
        if category == "required"
    )

    preferred_skills = sorted(
        skill
        for skill, category
        in category_map.items()
        if category == "preferred"
    )

    general_skills = sorted(
        skill
        for skill, category
        in category_map.items()
        if category == "general"
    )

    # ==================================================
    # RESULT
    # ==================================================

    return {
        "all_skills": sorted(
            set(
                all_detected_skills
            )
        ),

        "required_skills":
            required_skills,

        "preferred_skills":
            preferred_skills,

        "general_skills":
            general_skills,

        "skill_count": len(
            all_detected_skills
        ),

        "has_content": bool(
            normalized_text
        ),
    }


# ==================================================
# RESUME ↔ JOB COMPARISON
# ==================================================

def compare_resume_with_job(
    resume_analysis,
    job_analysis,
):

    # ==================================================
    # RESUME SKILLS
    # ==================================================

    resume_skills = set(
        resume_analysis.get(
            "skills",
            [],
        )
    )

    # ==================================================
    # JD SKILL CATEGORIES
    # ==================================================

    required_skills = set(
        job_analysis.get(
            "required_skills",
            [],
        )
    )

    preferred_skills = set(
        job_analysis.get(
            "preferred_skills",
            [],
        )
    )

    general_skills = set(
        job_analysis.get(
            "general_skills",
            [],
        )
    )

    all_jd_skills = (
        required_skills
        | preferred_skills
        | general_skills
    )

    # ==================================================
    # REQUIRED SKILL COMPARISON
    # ==================================================

    matched_required_skills = sorted(
        resume_skills
        & required_skills
    )

    missing_required_skills = sorted(
        required_skills
        - resume_skills
    )

    if required_skills:

        required_coverage = (
            len(
                matched_required_skills
            )
            / len(
                required_skills
            )
            * 100
        )

    else:

        required_coverage = None

    # ==================================================
    # PREFERRED SKILL COMPARISON
    # ==================================================

    matched_preferred_skills = sorted(
        resume_skills
        & preferred_skills
    )

    missing_preferred_skills = sorted(
        preferred_skills
        - resume_skills
    )

    if preferred_skills:

        preferred_coverage = (
            len(
                matched_preferred_skills
            )
            / len(
                preferred_skills
            )
            * 100
        )

    else:

        preferred_coverage = None

    # ==================================================
    # GENERAL SKILL COMPARISON
    # ==================================================

    matched_general_skills = sorted(
        resume_skills
        & general_skills
    )

    missing_general_skills = sorted(
        general_skills
        - resume_skills
    )

    # ==================================================
    # OVERALL DETECTED SKILL COMPARISON
    # ==================================================

    matched_skills = sorted(
        resume_skills
        & all_jd_skills
    )

    missing_skills = sorted(
        all_jd_skills
        - resume_skills
    )

    additional_skills = sorted(
        resume_skills
        - all_jd_skills
    )

    if all_jd_skills:

        overall_coverage = (
            len(
                matched_skills
            )
            / len(
                all_jd_skills
            )
            * 100
        )

    else:

        overall_coverage = None

    # ==================================================
    # RESULT
    # ==================================================

    return {

        # Existing dashboard compatibility fields

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "additional_skills":
            additional_skills,

        "matched_count":
            len(
                matched_skills
            ),

        # Legacy compatibility field.
        # Represents all detected JD skills.

        "required_count":
            len(
                all_jd_skills
            ),

        "match_percentage":
            overall_coverage,

        # Required skills

        "matched_required_skills":
            matched_required_skills,

        "missing_required_skills":
            missing_required_skills,

        "required_skill_count":
            len(
                required_skills
            ),

        "required_matched_count":
            len(
                matched_required_skills
            ),

        "required_coverage":
            required_coverage,

        # Preferred skills

        "matched_preferred_skills":
            matched_preferred_skills,

        "missing_preferred_skills":
            missing_preferred_skills,

        "preferred_skill_count":
            len(
                preferred_skills
            ),

        "preferred_matched_count":
            len(
                matched_preferred_skills
            ),

        "preferred_coverage":
            preferred_coverage,

        # General skills

        "matched_general_skills":
            matched_general_skills,

        "missing_general_skills":
            missing_general_skills,

        "general_skill_count":
            len(
                general_skills
            ),

        # Overall

        "overall_skill_count":
            len(
                all_jd_skills
            ),

        "overall_coverage":
            overall_coverage,
    }