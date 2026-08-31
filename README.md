# 🎓 Student Placement Intelligence System

A data-driven placement analytics and student intelligence platform that transforms student placement data into meaningful analytics, benchmark comparisons, personalized preparation recommendations, machine-learning estimates, and AI-generated placement-readiness reports.

## 🚀 Live Demo

**Try the deployed application:**

https://student-placement-intelligence-vaidehi.streamlit.app/

> **Important:** This project currently uses a **synthetic student placement dataset**. Analytics, recommendations, and machine-learning outputs are intended for demonstration and decision-support purposes and must not be interpreted as guaranteed real-world placement outcomes.

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

In the current synthetic dataset, these differences are relatively small. The dashboard therefore warns users not to over-interpret minor differences.

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

For example, a very small group showing a high placement rate is not automatically presented as a reliable general trend.

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

Depending on the student's profile, recommendations may include:

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

### Model Evaluation Results

| Model                | Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| -------------------- | -------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression  |   55.77% |    59.90% |     56.83% |     58.32% | **58.50%** |
| Random Forest        |   55.25% |    58.98% |     58.58% |     58.78% |     56.96% |
| Gradient Boosting    |   56.68% |    56.96% | **83.72%** | **67.80%** |     57.96% |
| Soft Voting Ensemble |   56.19% |    58.57% |     66.79% |     62.41% |     58.17% |

Logistic Regression produced the strongest ROC-AUC among the evaluated models and was selected for the saved prediction pipeline.

### Important ML Limitation

The predictive performance is intentionally reported transparently.

The dataset contains relatively weak predictive separation between placed and non-placed students. Therefore, this project does **not** claim that the model can reliably determine whether a real student will be placed.

The model output is presented as a:

> **Prototype placement estimate based on synthetic data**

rather than a guaranteed placement probability.

The trained model is cached using `joblib`, preventing unnecessary retraining whenever the application starts.

---

## ✨ AI-Generated Placement Readiness Reports

The project integrates the **Groq API** to generate natural-language student placement-readiness reports.

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

### AI Grounding and Guardrails

The LLM is instructed to:

* Use only supplied application facts.
* Avoid inventing student information.
* Avoid changing numerical values.
* Avoid causal placement claims.
* Avoid presenting ML estimates as guarantees.
* Avoid inventing companies, platforms, certifications, or target careers.
* Avoid inventing preparation timelines or numerical improvement targets.
* Respect already-identified student strengths.
* Base recommendations on system-generated preparation actions.
* Clearly disclose that the dataset is synthetic.

### AI Fallback System

The application includes a deterministic fallback report generator.

If the external AI service is unavailable or returns an unusable response, the system generates a structured report using verified analytics and recommendations instead of allowing the application to fail.

This improves the reliability of the deployed application.

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
* Placed-student benchmark comparison
* Strengths
* Priority improvement areas
* Personalized preparation recommendations
* Machine-learning placement estimate
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

### Generative AI

* Groq API
* GPT-OSS
* Prompt engineering
* Grounded report generation
* Deterministic fallback reporting

### Application

* Streamlit

### Configuration

* python-dotenv
* Streamlit Secrets

### Version Control & Deployment

* Git
* GitHub
* Streamlit Community Cloud

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

* 100,000 records
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

The project uses the dataset to demonstrate the architecture and functionality of a placement intelligence system.

**Dataset:** Student Placement Prediction Dataset 2026
**Source:** Kaggle
**Author:** Mansehaj Preet
**License:** CC0: Public Domain

---

## 📈 Current Dataset Overview

The current dataset contains:

* **Total Students:** 100,000
* **Placed Students:** 54,459
* **Not Placed:** 45,541
* **Placement Rate:** 54.46%
* **Average Salary of Placed Students:** 13.32 LPA
* **Highest Salary:** 20.44 LPA

These values describe the current synthetic dataset and should not be generalized to real institutions or student populations.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/varshavarada/student-placement-intelligence-system.git
cd student-placement-intelligence-system
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

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

For local development, create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file or API credentials to GitHub.

For Streamlit Community Cloud deployment, the Groq API key should be configured through **Streamlit Secrets** rather than stored in the repository.

---

## ▶️ Running the Project

### Run the Analysis Pipeline

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

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

https://student-placement-intelligence-vaidehi.streamlit.app/

Deployment configuration:

```text
Repository:
varshavarada/student-placement-intelligence-system

Branch:
main

Main file:
app/dashboard.py
```

The Groq API credential is configured securely through Streamlit's secrets management and is not stored in the GitHub repository.

---

## 🔒 Security Practices

The project follows several basic security practices:

* API keys are stored using environment variables or deployment secrets.
* `.env` is excluded from Git.
* Virtual environments and IDE-specific files are excluded from version control.
* Secrets are not hard-coded in Python source files.
* The LLM receives structured application information rather than unrestricted system access.
* A deterministic fallback report protects application availability when the external AI service fails.

For real institutional deployment, additional authentication, authorization, encryption, privacy review, audit logging, and student-data governance would be required.

---

## ⚠️ Current Limitations

1. The current dataset is synthetic.
2. ML predictive performance is limited.
3. Placement relationships shown by analytics are associations, not proof of causation.
4. The system currently benchmarks students against aggregate placed-student statistics rather than institution-specific cohorts.
5. AI generation depends on an external LLM service when live generation is enabled.
6. The prototype does not currently include user authentication or institutional access controls.
7. Recommendations are preparation guidance rather than guarantees of placement success.
8. Real institutional deployment would require validated data, privacy controls, fairness evaluation, and continuous monitoring.

---

## 🔮 Future Improvements

Potential production-level extensions include:

* Integration with verified institutional placement data
* Department-specific benchmarking
* Batch-specific benchmarking
* Role-specific placement-readiness analysis
* Historical placement trend analysis
* Recruiter and company-level analytics
* Explainable machine-learning components
* Model monitoring and retraining
* Authentication and role-based access control
* Separate student and placement-officer dashboards
* Database integration
* Automated data ingestion
* Privacy and consent controls
* Institution-specific recommendation engines
* Longitudinal student progress tracking
* Model fairness and bias monitoring

---

## 💡 Project Philosophy

The objective of this project is not simply to predict whether a student will be placed.

Its primary goal is to convert student placement data into **understandable and actionable intelligence**.

The system combines:

**Data Analytics + Statistical Benchmarking + Student Intelligence + Machine Learning + Generative AI + Interactive Visualization**

to support more informed placement preparation.

Machine learning is only one component of the overall system.

The primary value comes from combining transparent analytics, benchmark-based student intelligence, personalized recommendations, and grounded natural-language reporting.

---

## ⚖️ Responsible Use

This project should be treated as a **decision-support prototype**.

Placement decisions should never be made solely from the ML estimate, benchmark score, or AI-generated report.

Real-world institutional deployment would require:

* Validated real-world data
* Fairness and bias evaluation
* Student privacy protection
* Appropriate consent mechanisms
* Model monitoring
* Human oversight
* Secure authentication and authorization
* Institutional data governance

---

## 👩‍💻 Repository

GitHub Repository:

https://github.com/varshavarada/student-placement-intelligence-system

Live Application:

https://student-placement-intelligence-vaidehi.streamlit.app/

---

## 📄 License and Dataset Attribution

The project source code may be licensed according to the repository owner's chosen software license.

The dataset used by this prototype is separately distributed as:

**Student Placement Prediction Dataset 2026**
**Author:** Mansehaj Preet
**Platform:** Kaggle
**License:** CC0: Public Domain

The dataset is used for prototype development and demonstration purposes.
