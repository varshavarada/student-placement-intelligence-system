# 🎓 Student Placement Intelligence System

A data-driven placement analytics and student intelligence platform that transforms student placement data into meaningful analytics, benchmark comparisons, personalized preparation recommendations, machine-learning estimates, and AI-generated placement-readiness reports.

> **Important:** This project currently uses a **synthetic student placement dataset**. Analytics and machine-learning outputs are intended for demonstration and decision-support purposes and must not be interpreted as guaranteed real-world placement outcomes.

---

## 📌 Problem Statement

Educational institutions collect academic, aptitude, technical, communication, project, internship, and placement-related student data.

However, raw student data alone does not provide clear answers to questions such as:

* What patterns exist among placed and non-placed students?
* Which areas of a student's profile are stronger or weaker compared with placement benchmarks?
* What preparation areas should a student prioritize?
* How can placement information be presented in an understandable and actionable form?

Manually analyzing large student datasets is also time-consuming and makes consistent student-level evaluation difficult.

The **Student Placement Intelligence System** addresses this problem by transforming placement data into analytics, benchmark-based student intelligence, personalized recommendations, machine-learning estimates, and grounded AI-generated reports.

---

## 🎯 Project Objectives

The system aims to:

* Analyze overall placement performance.
* Compare placed and non-placed student profiles.
* Identify placement-related patterns in the dataset.
* Analyze salary distributions for placed students.
* Benchmark individual students against placed-student averages.
* Identify meaningful strengths and priority improvement areas.
* Generate personalized preparation recommendations.
* Produce a prototype machine-learning placement estimate.
* Generate natural-language placement-readiness reports using an LLM.
* Present results through an interactive Streamlit dashboard.

---

## 🏗️ System Architecture

```text
Student Placement Dataset
          │
          ▼
    Data Ingestion
          │
          ▼
Data Validation & Analysis
          │
          ├──────────────► Placement Analytics
          │
          ├──────────────► Salary Analytics
          │
          └──────────────► Experience Trend Analysis
          │
          ▼
 Student Benchmarking
          │
          ▼
Strength / Improvement Detection
          │
          ▼
Personalized Recommendations
          │
          ├──────────────► Machine Learning Estimate
          │
          └──────────────► Grounded LLM Report
          │
          ▼
   Streamlit Dashboard
```

The architecture deliberately separates deterministic analytics from generative AI.

The LLM does not independently calculate placement statistics. It receives already-computed student information, benchmark comparisons, recommendations, and the ML estimate and converts them into a readable placement-readiness report.

---

## ✨ Core Features

### 1. Placement Analytics

The system calculates:

* Total number of students
* Number of placed students
* Number of non-placed students
* Overall placement rate
* Average salary package
* Highest salary package

---

### 2. Placed vs Non-Placed Comparison

The system compares average values across attributes such as:

* CGPA
* Coding skill
* Aptitude
* Communication
* Logical reasoning
* Mock interview performance
* Projects
* Internships

This helps identify differences present in the dataset without claiming that those differences cause placement outcomes.

---

### 3. Branch and College-Tier Analysis

Placement rates are analyzed across:

* Academic branches
* College tiers

In the current synthetic dataset, these differences are relatively small. The dashboard therefore explicitly warns users not to over-interpret minor differences.

---

### 4. Salary Analytics

For placed students, the system calculates:

* Mean salary
* Median salary
* Minimum salary
* Maximum salary
* 25th percentile
* 75th percentile

A salary distribution visualization is also provided.

---

### 5. Internship and Project Trend Analysis

The system analyzes placement rates across different internship and project counts.

To reduce misleading conclusions from very small groups, each group includes:

* Sample size
* Number of placed students
* Placement rate
* Reliability flag

Groups containing fewer than **100 students** are treated as low-sample groups and excluded from the main trend visualization.

For example, a very small group showing a 100% placement rate is not presented as a reliable general trend.

These analyses represent **associations within the synthetic dataset and not causal relationships**.

---

## 🧠 Student Intelligence

The system provides student-level analysis by comparing an individual student's profile with benchmarks calculated from placed students.

The comparison includes:

* CGPA
* Coding skill
* Aptitude
* Communication
* Logical reasoning
* Mock interview performance
* Projects
* Internships
* Certifications
* GitHub repositories

### Benchmark Method

For each feature, the system calculates a standardized difference relative to the placed-student benchmark.

To prevent insignificant differences from being presented as meaningful:

```text
Standardized Difference >= +0.25 → Strength

Standardized Difference <= -0.25 → Improvement Area

Between -0.25 and +0.25 → Neutral
```

This creates a neutral zone around the benchmark and prevents very small differences from being unnecessarily classified.

---

## 🎯 Personalized Recommendations

Recommendations are generated from the student's identified improvement areas.

Examples include:

* Improving coding through practical development
* Practicing quantitative aptitude
* Strengthening communication
* Practicing logical reasoning
* Increasing mock interview preparation
* Improving project documentation
* Gaining practical internship exposure
* Publishing and documenting existing projects on GitHub
* Completing relevant certifications

Recommendations are designed to respect existing strengths.

For example, if project experience is already a strength but GitHub presence is weak, the system prioritizes publishing and documenting existing projects instead of automatically recommending more projects.

---

## 🤖 Machine Learning Component

The project evaluates multiple machine-learning approaches for placement estimation.

Models evaluated include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Soft Voting Ensemble

### Evaluation Results

| Model                | Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| -------------------- | -------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression  |   55.77% |    59.90% |     56.83% |     58.32% | **58.50%** |
| Random Forest        |   55.25% |    58.98% |     58.58% |     58.78% |     56.96% |
| Gradient Boosting    |   56.68% |    56.96% | **83.72%** | **67.80%** |     57.96% |
| Soft Voting Ensemble |   56.19% |    58.57% |     66.79% |     62.41% |     58.17% |

Logistic Regression produced the strongest ROC-AUC among the evaluated models and was selected for the saved prediction pipeline.

### Important ML Limitation

The predictive performance is intentionally reported transparently.

The dataset contains relatively weak predictive separation between placed and non-placed students. Therefore, the project does **not** claim that the model can reliably determine whether a real student will be placed.

The model output is presented as a:

> **Prototype placement estimate based on synthetic data**

rather than a guaranteed placement probability.

The trained model is cached using `joblib`, preventing unnecessary retraining whenever the application starts.

---

## ✨ AI-Generated Placement Readiness Reports

The project integrates the Groq API to generate natural-language student placement-readiness reports.

The current report generation pipeline uses:

```text
Verified Student Data
        +
Placed-Student Benchmarks
        +
Detected Strengths
        +
Improvement Areas
        +
System Recommendations
        +
ML Estimate
        │
        ▼
Grounded Prompt
        │
        ▼
Groq LLM
        │
        ▼
Placement Readiness Report
```

The generated report contains:

* Placement Readiness Summary
* Key Strengths
* Priority Improvement Areas
* Recommended Preparation Plan
* Model Estimate
* Important Note

### AI Grounding and Safety

The LLM is instructed to:

* Use only supplied application facts.
* Avoid inventing student information.
* Avoid changing numerical values.
* Avoid causal placement claims.
* Avoid presenting ML estimates as guarantees.
* Avoid inventing companies, platforms, certifications, or target careers.
* Avoid inventing timelines or preparation schedules.
* Respect already-identified student strengths.
* Clearly disclose that the dataset is synthetic.

If the Groq service fails or returns an unusable response, the application automatically generates a deterministic fallback report instead of crashing.

---

## 📊 Interactive Dashboard

The application is built using **Streamlit** and contains two primary sections.

### Placement Overview

Provides:

* Overall placement metrics
* Placement-status distribution
* Branch-wise placement analysis
* College-tier analysis
* Internship placement trends
* Project placement trends
* Low-sample group inspection
* Salary statistics
* Salary distribution

### Student Intelligence

Allows a student ID to be entered and displays:

* Student profile
* Benchmark comparison
* Strengths
* Improvement areas
* Personalized recommendations
* ML placement estimate
* AI-generated placement-readiness report

---

## 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Machine Learning

* Scikit-learn
* Joblib

### AI / LLM

* Groq API
* GPT-OSS model
* Prompt grounding and guardrails

### Application

* Streamlit

### Configuration

* python-dotenv

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
csv_project/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── placement_data.csv
│
├── models/
│   └── placement_model.joblib
│
├── outputs/
│   └── charts/
│       ├── placement_distribution.png
│       ├── branch_placement_rate.png
│       └── salary_distribution.png
│
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── analytics.py
│   ├── visualization.py
│   ├── intelligence.py
│   ├── ml_model.py
│   └── report_generator.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📚 Dataset

The prototype uses the **Student Placement Prediction Dataset 2026** published on Kaggle by Mansehaj Preet.

Dataset characteristics:

* Approximately 100,000 records
* 26 columns
* Academic attributes
* Technical skill scores
* Aptitude and reasoning scores
* Communication and mock interview scores
* Project and internship experience
* Certifications
* GitHub activity
* Placement status
* Salary package

### Dataset Disclaimer

The dataset is **synthetic**.

It does not represent 100,000 verified real-world students.

The project uses it to demonstrate the architecture and functionality of a placement intelligence system.

Dataset source:

`Student Placement Prediction Dataset 2026 — Kaggle`

License:

`CC0: Public Domain`

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file or API keys to GitHub.

The repository's `.gitignore` should exclude sensitive configuration.

---

## 📥 Dataset Setup

Download the dataset from Kaggle and place the CSV file at:

```text
data/placement_data.csv
```

The dataset file is intentionally excluded from Git version control.

---

## ▶️ Running the Project

### Run the Complete Analysis Pipeline

```bash
python src/main.py
```

This executes the analytics pipeline, loads or trains the ML component where required, and generates static visualizations.

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Streamlit will provide a local application URL in the terminal.

---

## 🔒 Security Practices

The project follows several basic security practices:

* API keys are stored through environment variables.
* `.env` is excluded from Git.
* The dataset is excluded from the repository.
* The LLM receives structured application information rather than unrestricted system access.
* A deterministic fallback report protects application availability when the external AI service fails.

For real institutional deployment, additional authentication, authorization, encryption, privacy review, logging, and student-data governance would be required.

---

## ⚠️ Current Limitations

1. The current dataset is synthetic.
2. ML predictive performance is limited.
3. Placement relationships shown by analytics are associations, not proof of causation.
4. The system currently benchmarks students against aggregate placed-student statistics rather than institution-specific cohorts.
5. The AI report depends on an external LLM service when live generation is enabled.
6. The prototype does not currently include user authentication or institutional access controls.
7. Recommendations are preparation guidance rather than guarantees of placement success.

---

## 🔮 Future Improvements

Potential production-level extensions include:

* Integration with verified institutional placement data
* Department and batch-specific benchmarking
* Role-specific placement readiness analysis
* Historical placement trend analysis
* Recruiter and company-level analytics
* Explainable ML components
* Model monitoring and retraining
* Authentication and role-based access control
* Student and placement-officer dashboards
* Database integration
* Automated data ingestion
* Privacy and consent controls
* Institution-specific recommendation engines
* Longitudinal student progress tracking

---

## 💡 Project Philosophy

The objective of this project is not simply to predict whether a student will be placed.

Its primary goal is to convert student placement data into **understandable and actionable intelligence**.

The system combines:

**Data Analytics + Statistical Benchmarking + Student Intelligence + Machine Learning + Generative AI + Interactive Visualization**

to support more informed placement preparation.

---

## ⚖️ Responsible Use

This project should be treated as a decision-support prototype.

Placement decisions should never be made solely from the ML estimate, benchmark score, or AI-generated report.

Real-world deployment would require validated institutional data, fairness evaluation, privacy protection, model monitoring, human oversight, and appropriate student-data governance.

---

## 📄 License

The project source code may be licensed according to the repository owner's chosen software license.

The dataset used by the prototype is separately distributed under its stated **CC0 Public Domain** license.
