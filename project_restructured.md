# Mission Overview: Model Scoring Production Deployment

## 1. General Mission Context

This mission simulates the **production deployment of a scoring model**.
To achieve this, it is necessary to:

*   Carefully read the mission brief.
*   Consult the various steps.
*   Prepare your questions for the mentoring session.

### 1.1 Scenario

You are a **Data Scientist** at the company **"Prêt à Dépenser"**.

After developing and versioning a scoring model during the project **"Introduction to MLOps" (Part 1/2)**, you receive a Slack message from **Chloé Dubois**, Lead Data Scientist:

> *"Hi! Excellent results on the latest version of the scoring model!
> The "Crédit Express" department is very eager to use it to process new requests in near real-time.
> We absolutely need a functional and deployable API (Docker Ready!) by the end of next week.
> Can you prioritize this?
> We also need a dashboard or monitoring report to check that everything is going well once in production (score distribution, response time, that kind of thing).
> Keep me informed of your action plan!
> Thanks!"*

### 1.2 Role and Responsibilities

You are responsible for **piloting the effective production deployment of the scoring model**, which includes:

*   Creating a **robust API**.
*   **Containerization** for smooth deployment.
*   Implementing **proactive monitoring**.
*   Ensuring the **performance and reliability** of the model over time.

## 2. Project Deliverables

### 2.1 Version History

*   History tracing the construction of the project.
*   Available via the GitHub commit list.

### 2.2 Scripts

#### 2.2.1 Functional API

*   API developed with **Gradio or FastAPI**.
*   Input: customer data.
*   Output: prediction score.

#### 2.2.2 Automated Unit Tests

#### 2.2.3 Dockerfile

*   Complete code containerization.

### 2.3 Data Drift Analysis

#### 2.3.1 Monitoring and Visualization

*   Dashboard or report (Notebook, Streamlit, Dash).
*   Key metrics:
    *   Distribution of predicted scores.
    *   API latency.
    *   Inference time.

#### 2.3.2 Data Storage

*   Screenshots of the production data storage solution.

### 2.4 CI/CD Pipeline

*   YAML file (or equivalent).
*   Automation:
    *   Tests.
    *   Production deployment.
*   Triggered at a minimum on a push to the main branch.

### 2.5 Documentation

*   **README** file.
*   Explanations:
    *   API launch.
    *   Monitoring interpretation.

## 3. Reutilization of Previous Project

This project builds upon the deliverables of the project:

> **"Introduction to MLOps (Part 1/2)"**

You must:

*   Re-use the already developed scoring model.
*   Re-use the MLflow artifacts.
*   Adapt elements if necessary.
*   Build a complete deployment environment around the model.

## 4. Recommended Tools

*   **Streamlit**
*   **Gradio**

You may use other tools, provided you **justify your technical choices during the defense**.

# Project Steps

## 5. Step 1 — Git Repository Initialization

### 5.1 Description

*   Initialize a Git repository.
*   Clearly structure the project:
    *   Source code.
    *   Tests.
    *   Notebooks.
    *   Dockerfile.
    *   Requirements.
*   Add the model, inference scripts, analysis notebooks, and documentation.
*   Push the repository to a remote platform.

### 5.2 Prerequisites

*   Git installed.
*   GitHub / GitLab / Bitbucket account.

### 5.3 Expected Results

*   Link to a public Git repository.
*   Clear commit history.

### 5.4 Recommendations

*   Explicit commit messages.
*   Branching strategy if necessary.
*   `.gitignore` file.

### 5.5 Points of Attention

*   Never commit sensitive data.
*   Repository must be public.

### 5.6 Tools

*   Git.
*   GitHub / GitLab / Bitbucket.

### 5.7 Resources

*   Git documentation.
*   GitHub Quickstart.
*   Course "Manage code with Git and GitHub".

## 6. Step 2 — API, Docker, and CI/CD

### 6.1 Description

*   Develop an API (Gradio or FastAPI).
*   Containerize the API with Docker.
*   Create an automated CI/CD pipeline:
    1.  Execute tests.
    2.  Build the Docker image.
    3.  Deploy the API.

### 6.2 Prerequisites

*   Versioned code.
*   Chosen API framework.
*   Docker installed.

### 6.3 Expected Results

*   Functional API.
*   Dockerfile.
*   Operational CI/CD pipeline.
*   Integrated automated tests.

### 6.4 Recommendations

*   Start simple and iterate.
*   Error management and documentation (Swagger).
*   Separation of build / test / deployment.
*   Use of secrets.
*   Recommended use: **Hugging Face Spaces**.

### 6.5 Points of Attention

*   Tests covering critical cases:
    *   Missing data.
    *   Outliers.
    *   Incorrect types.
*   API security.
*   Model loading **only once at startup**.
*   Sufficient resources in the environment.

### 6.6 Tools

*   Gradio / FastAPI.
*   Docker.
*   Postman / curl.
*   GitHub Actions / GitLab CI / Jenkins.
*   Pytest.
*   Deployment platforms.

### 6.7 Resources

*   FastAPI documentation.
*   Gradio documentation.
*   Docker.
*   Automated tests.
*   GitHub Actions.

## 7. Step 3 — Monitoring and Data Drift

### 7.1 Description

*   Storage of production data:
    *   Logs.
    *   Inputs / outputs.
    *   Execution times.
*   Automatic analysis:
    *   Drift detection.
    *   Operational anomalies.

### 7.2 Prerequisites

*   API deployed.
*   Data to be logged identified.

### 7.3 Expected Results

*   Described or implemented storage solution.
*   Analysis script or notebook.
*   Presentation of results and points of attention.

### 7.4 Recommendations

*   Structured logging (JSON).
*   Drift libraries.
*   Visualization via dashboard.

### 7.5 Points of Attention

*   Cost and storage.
*   GDPR.
*   Necessary reference data.

### 7.6 Tools

*   Python Logging.
*   Fluentd / Logstash.
*   Elasticsearch / PostgreSQL.
*   Evidently AI / NannyML.
*   Grafana / Kibana / Dash / Streamlit.

### 7.7 Resources

*   ML Monitoring in Python article.
*   Evidently documentation.

## 8. Step 4 — Post-Deployment Optimization

### 8.1 Description

*   Analysis of production performance.
*   Identification of bottlenecks.
*   Optimizations:
    *   Quantization.
    *   Code optimization.
    *   Hardware.
*   Deployment via CI/CD.
*   Documentation of results.

### 8.2 Prerequisites

*   API deployed.
*   Active monitoring.

### 8.3 Expected Results

*   Detailed optimization report.
*   Optimized version deployed.
*   Justification of choices.
*   Measured performance improvement.

### 8.4 Recommendations

*   Based on monitoring data.
*   Rigorous documentation.

### 8.5 Points of Attention

*   No regression (precision, bias).
*   Environment compatibility.

### 8.6 Tools

*   cProfile.
*   ONNX Runtime.

### 8.7 Resources

*   cProfile documentation.
*   ONNX Runtime documentation.

# Final Verification

## 9. Final Steps

*   Complete the **self-assessment sheet**.
*   Check in with your mentor.
*   Discuss during the last mentoring session.

---

Would you now like:

*   an **actionable checklist**
*   a **ready-to-deliver README**
*   a **defense plan**
*   or an **MLOps requirements ↔ tools mapping**?