# Student Placement Intelligence System

A Streamlit-based placement analytics project built to make student placement data easier to understand and use.

The system analyzes placement data, compares individual students with placement benchmarks, highlights areas that may need improvement, gives preparation suggestions, supports institutional dataset uploads, answers placement-related questions through a chatbot, and compares resume skills with job-description requirements.

## Live Demo

https://student-placement-intelligence-vaidehi.streamlit.app/

> The default dataset used in this project is synthetic. The results are intended for project demonstration and analysis and should not be treated as guaranteed real-world placement outcomes.

---

## Why I Built This Project

Colleges collect a lot of student information such as CGPA, aptitude scores, coding skills, communication scores, projects, internships and placement results.

The problem is that collecting this information does not automatically make it useful.

It can still be difficult to understand:

* how placed and non-placed students differ,
* where an individual student stands compared with placed students,
* which areas a student should focus on,
* what trends are visible in placement data,
* and how well a student's skills match a particular job description.

I built this project to bring these different tasks together in one application instead of treating placement data as only a collection of rows and columns.

---

## Main Features

The application currently has four main sections:

* **Overview**
* **Student Intelligence**
* **Resume & Job Match**
* **Placement Copilot**

It also supports uploading institution datasets in addition to the built-in demo dataset.

---

## Overview

The Overview page provides dataset-level placement analysis.

Depending on the columns available in the active dataset, it can show:

* total students,
* placed and non-placed students,
* placement rate,
* salary statistics,
* placement-status distribution,
* branch-wise analysis,
* college-tier analysis,
* internship trends,
* project trends,
* and placed vs non-placed comparisons.

The dashboard does not treat every difference as an important finding. Small differences are shown carefully because they may not represent meaningful patterns.

The trends shown here describe associations in the dataset. They do not prove that a particular feature caused a placement outcome.

### Low-Sample Groups

Internship and project analysis also considers group size.

Groups containing fewer than **100 students** are treated as low-sample groups and are not used as strong trend evidence.

This helps avoid misleading conclusions from very small groups.

---

## Student Intelligence

The Student Intelligence page focuses on an individual student.

After selecting a student ID, the system can show:

* student profile,
* comparison with placed-student benchmarks,
* strengths,
* improvement areas,
* preparation recommendations,
* a machine-learning estimate for the demo dataset,
* and a placement-readiness report.

### Benchmarking Logic

Student values are compared with averages calculated from placed students.

```text
Difference >= +0.25  → Strength

Difference <= -0.25  → Improvement Area

Between -0.25 and +0.25  → Neutral
```

The neutral range prevents very small differences from being labelled as strengths or weaknesses.

### Preparation Recommendations

Recommendations are based on the improvement areas identified for the student.

Depending on the profile, they may focus on:

* coding,
* aptitude,
* communication,
* logical reasoning,
* mock interviews,
* projects,
* internships,
* certifications,
* and GitHub activity.

The recommendation logic also considers existing strengths.

For example, if project experience is already strong but GitHub activity is weak, the system can suggest documenting and publishing existing projects instead of simply recommending more projects.

---

## Institution Dataset Upload

The application is not limited to the built-in demo dataset.

Users can choose between:

```text
Demo Dataset
Upload Institution Data
```

Currently supported upload formats are:

* CSV
* XLSX

### Column Mapping

Different colleges may use different names for the same information, so the application includes a flexible mapping process.

It supports:

* automatic column detection,
* manual mapping,
* required-field checks,
* optional fields,
* duplicate mapping checks,
* salary-unit selection,
* and validation before the dataset becomes active.

The minimum required fields are:

```text
Student ID
Placement Status
```

Other analytical fields are optional.

If a field required for a particular analysis is missing, that feature is disabled instead of guessing a value.

### Salary Handling

Uploaded datasets may store salary in different units.

The application currently supports:

* LPA,
* Annual INR,
* Monthly INR.

The user selects the correct unit instead of the system making an automatic assumption.

### ML Safety for Uploaded Data

The saved machine-learning model was trained on the synthetic demo dataset.

Because another institution's data may have a different distribution, I do not automatically apply the demo-trained model to uploaded institutional datasets.

I preferred disabling the prediction in that case rather than showing a number that could be misleading.

---

## Machine Learning

I tested four models for placement estimation:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Soft Voting Ensemble

### Model Results

| Model                | Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| -------------------- | -------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression  |   55.77% |    59.90% |     56.83% |     58.32% | **58.50%** |
| Random Forest        |   55.25% |    58.98% |     58.58% |     58.78% |     56.96% |
| Gradient Boosting    |   56.68% |    56.96% | **83.72%** | **67.80%** |     57.96% |
| Soft Voting Ensemble |   56.19% |    58.57% |     66.79% |     62.41% |     58.17% |

Logistic Regression gave the highest ROC-AUC among the models I tested, so it was selected for the saved prediction pipeline.

### About the ML Result

The model performance is not very strong, and I have kept that visible intentionally.

The current synthetic dataset does not provide strong predictive separation between placed and non-placed students.

Because of this, the model output is shown only as a:

> **Prototype placement estimate based on synthetic data**

It should not be interpreted as the real probability that a student will get placed.

---

## Placement Readiness Report

The project uses the Groq API to generate a readable placement-readiness report.

The LLM does **not** calculate the placement statistics.

The application first calculates the student's results using the analytics, benchmarking and recommendation modules.

The report is then generated from:

```text
Student Data
+
Benchmark Comparison
+
Strengths
+
Improvement Areas
+
Recommendations
+
ML Estimate
```

The LLM is mainly used to explain these already-calculated results in natural language.

If the external AI service is unavailable, the application can generate a fallback report using the information already calculated by the system.

---

## Placement Copilot

Placement Copilot is the chatbot inside the application.

It can handle:

* questions about the active dataset,
* questions about a particular student,
* placement-preparation questions,
* questions about the application,
* general conversation,
* and follow-up questions.

For questions that require dataset statistics, the application calculates the required information from the active dataset instead of expecting the chatbot to invent numbers.

### Multilingual Support

Placement Copilot currently supports:

* English
* Tamil
* Hindi
* Kannada

It also handles common Romanized forms, which means users can type Indian-language sentences using English characters.

---

## Resume & Job Match

The Resume & Job Match module compares skills detected in a resume with skills mentioned in a job description.

Users can either upload a document or paste the text manually.

Supported document types:

* PDF
* DOCX
* TXT

DOCX upload has been tested through the current application workflow.

PDF and TXT support are implemented, but not every possible document layout has been tested.

### Skill Detection

The current implementation uses a predefined skill list with aliases.

Some supported skills include:

* Python
* Java
* JavaScript
* HTML
* CSS
* SQL
* MySQL
* PostgreSQL
* MongoDB
* React
* Node.js
* Flask
* Django
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Plotly
* Scikit-learn
* Machine Learning
* Deep Learning
* Data Analysis
* Data Visualization
* Statistics
* NLP
* LLMs
* RAG
* Git
* GitHub
* Docker
* AWS
* Azure
* Power BI
* Tableau
* Excel
* Communication
* Leadership
* Problem Solving

The current matcher is rule-based and deterministic. It is not a full semantic resume parser.

### Job Description Classification

Skills detected in the job description are grouped into:

* **Required Skills**
* **Preferred Skills**
* **Other Mentioned Skills**

The classification uses section headings and wording such as required, mandatory, preferred, desirable and similar terms.

### Skill Coverage

The module displays:

* Required Skill Coverage
* Preferred Skill Coverage
* Overall Detected Skill Coverage
* Matched Required Skills
* Missing Required Skills
* Matched Preferred Skills
* Missing Preferred Skills
* Other Mentioned Skills
* Additional Resume Skills

The percentage shown here represents **detected job-skill coverage**.

It is not a hiring probability, interview probability, ATS score, recruiter score or placement probability.

---

## Technology Stack

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

### AI

* Groq API
* GPT-OSS

### Document Processing

* pypdf
* python-docx

### Application

* Streamlit

### Other Tools

* python-dotenv
* Git
* GitHub
* Streamlit Community Cloud

---

## Project Structure

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
│
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── schema_mapper.py
│   ├── analytics.py
│   ├── visualization.py
│   ├── intelligence.py
│   ├── ml_model.py
│   ├── report_generator.py
│   ├── chatbot.py
│   ├── career_intelligence.py
│   └── document_parser.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset

The default dataset used in this project is:

**Student Placement Prediction Dataset 2026**

* **Author:** Mansehaj Preet
* **Platform:** Kaggle
* **License:** CC0: Public Domain
* **Rows:** 100,000
* **Columns:** 26

The dataset contains academic, technical, aptitude, communication, internship, project, placement and salary-related information.

### Important Note

The dataset is synthetic.

The 100,000 records should not be described as data collected from 100,000 real students.

I use the dataset to develop and demonstrate the placement-intelligence workflow.

### Demo Dataset Statistics

* **Total Students:** 100,000
* **Placed Students:** 54,459
* **Not Placed Students:** 45,541
* **Placement Rate:** 54.46%
* **Average Salary of Placed Students:** 13.32 LPA
* **Highest Salary:** 20.44 LPA

These values describe only the bundled synthetic dataset.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/varshavarada/student-placement-intelligence-system.git
cd student-placement-intelligence-system
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

For local use, create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or API keys to GitHub.

For Streamlit deployment, the API key can be stored using Streamlit Secrets.

---

## Running the Project

Run the main pipeline:

```bash
python src/main.py
```

Start the Streamlit application:

```bash
streamlit run app/dashboard.py
```

---

## Deployment

The project is deployed using Streamlit Community Cloud.

**Live Application**

https://student-placement-intelligence-vaidehi.streamlit.app/

**GitHub Repository**

https://github.com/varshavarada/student-placement-intelligence-system

**Branch**

```text
main
```

**Main Streamlit File**

```text
app/dashboard.py
```

---

## Current Limitations

The current version still has some limitations:

* the built-in dataset is synthetic,
* the ML model has limited predictive performance,
* placement trends show associations rather than causation,
* the demo-trained model is not applied directly to unrelated institution datasets,
* some dashboard features require specific columns,
* student benchmarking currently uses overall placed-student averages,
* live AI features depend on an external API,
* resume/JD matching uses a predefined supported-skill list,
* semantic skill matching is not implemented,
* OCR is not available for scanned or image-only documents,
* complex PDF layouts may not always extract correctly,
* resume skill coverage is not an ATS score,
* authentication and role-based access are not part of the current version,
* and real-world fairness testing has not yet been carried out.

---

## Future Enhancements

There are several useful directions in which the project can be extended.

### Better Institution-Level Analysis

* real institutional placement-data integration,
* department-wise benchmarking,
* batch-wise benchmarking,
* historical placement analysis,
* institution-specific recommendations.

### Role and Company Matching

* role-specific skill analysis,
* company-specific skill-gap analysis,
* company eligibility checking,
* job-role recommendations,
* recruiter/company-level analytics.

### Resume Matching Improvements

* semantic skill matching,
* embedding-based similarity,
* larger skill taxonomy,
* better synonym handling,
* experience requirement matching,
* education requirement matching,
* improved resume/JD section detection.

### Better Document Support

* OCR for scanned resumes,
* image-based PDF support,
* better multi-column extraction,
* improved table handling,
* support for additional document types.

### Student Progress Tracking

A future version could track student improvement over time, including:

* aptitude,
* coding,
* mock interviews,
* projects,
* certifications,
* resume updates,
* and placement-readiness progress.

### Machine Learning Improvements

If validated real institutional data becomes available, the ML component can be extended with:

* institution-specific training,
* stronger validation,
* explainable ML,
* feature importance,
* model calibration,
* retraining,
* performance monitoring,
* and data-drift checks.

### Placement Copilot Improvements

Future improvements may include:

* more advanced dataset questions,
* additional languages,
* resume/JD explanations through chat,
* institution-specific knowledge,
* evidence-backed responses,
* and voice interaction.

### Institutional Features

A larger institutional version could include:

* student login,
* faculty login,
* placement-officer login,
* admin access,
* role-based permissions,
* database integration,
* profile management,
* audit logs,
* and controlled report exports.

---

## Project Purpose

The main aim of this project is not simply to predict whether a student will get placed.

I wanted to make placement data more useful.

The project helps answer questions such as:

* What patterns are present in the placement data?
* Where does an individual student stand?
* Which areas may need more preparation?
* What skills are missing for a particular job description?
* How can this information be presented in a simpler form?

Machine learning is one part of the project, not the complete project.

The larger goal is to combine analytics, benchmarking, preparation guidance, resume matching and conversational access in one application.

---

## Responsible Use

This project is currently a prototype.

The ML estimate, benchmark comparison, placement-readiness report, chatbot responses and resume/JD skill coverage should not be used alone to make real recruitment or student decisions.

A production version would require verified institutional data, privacy controls, secure access, fairness testing and human oversight.

---

## Repository

**GitHub**

https://github.com/varshavarada/student-placement-intelligence-system

**Live Demo**

https://student-placement-intelligence-vaidehi.streamlit.app/

---

## Dataset Attribution

**Student Placement Prediction Dataset 2026**
Author: Mansehaj Preet
Platform: Kaggle
License: CC0: Public Domain

The dataset is used for project development and demonstration.
