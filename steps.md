# ModelGuard — Development Command Reference

**Project:** ModelGuard ML Reliability / MLOps Platform  
**Environment:** Windows PowerShell + Python 3.10  
**Backend:** FastAPI  
**Testing:** pytest  
**Experiment Tracking:** MLflow  
**Containerization:** Docker

---

## 1. Open the Project and Activate the Environment

Open the ModelGuard folder in VS Code and select the PowerShell terminal.

```powershell
cd "$HOME\OneDrive\Desktop\ModelGuard"
```

Activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
python -m pip --version
```

**Expected:** The terminal prompt starts with `(.venv)`.

> If `.venv` does not exist, create it once with `python -m venv .venv`. Do not recreate it every time you open the project.

---

## 2. Install or Verify Python Libraries

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the core data science libraries if needed:

```powershell
python -m pip install pandas numpy scikit-learn
```

Install or verify MLflow:

```powershell
python -m pip install mlflow
python -m pip show mlflow
```

Check FastAPI and Uvicorn:

```powershell
python -m pip show fastapi
python -m pip show uvicorn
```

Install them if they are missing:

```powershell
python -m pip install fastapi uvicorn
```

Install pytest if needed:

```powershell
python -m pip install pytest
```

---

## 3. ModelGuard Project Structure

The project contains these important directories:

```text
ModelGuard/
├── app/                 # FastAPI application
├── data/                # Dataset files
├── models/              # Saved trained model
├── notebooks/           # Experiments and exploration
├── reports/             # Metrics and evaluation reports
├── src/                 # Data processing and ML pipeline
├── tests/                # Automated tests
├── .venv/               # Virtual environment
├── .gitignore
├── Dockerfile
├── docker-compose.yml   # If configured
├── requirements.txt
└── steps.txt             # This command reference
```

Check the current project files before creating anything new.

To create a missing directory:

```powershell
New-Item -ItemType Directory -Force -Path data, src, tests, models, notebooks, reports, app
```

Create a missing file only when required:

```powershell
New-Item src\preprocessing.py -ItemType File
New-Item src\train.py -ItemType File
New-Item src\evaluate.py -ItemType File
```

---

## 4. Run the Machine Learning Pipeline

Run these commands from the ModelGuard project root with `.venv` activated.

### Step 1 — Data preprocessing

```powershell
python -m src.preprocessing
```

### Step 2 — Train the model

```powershell
python -m src.train
```

### Step 3 — Evaluate the model

```powershell
python -m src.evaluate
```

> Run only the modules that actually exist and are implemented in the current project. Check the relevant script if a command fails.

### Important ModelGuard outputs

- Trained model: `models/modelguard_model.pkl`
- Evaluation metrics: `reports/metrics.json`
- MLflow experiment: `ModelGuard-Income-Classification`

These paths should match the current code configuration.

---

## 5. Start the FastAPI Backend

### Step 1 — Verify dependencies

```powershell
python -m pip show fastapi
python -m pip show uvicorn
```

### Step 2 — Start the server

```powershell
python -m uvicorn app.main:app --reload
```

Keep this terminal open while testing the API.

### Step 3 — Open the API

- Main endpoint: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Use `/docs` to test the prediction endpoint and inspect request validation errors.

### Step 4 — Stop the server

Press:

```text
Ctrl + C
```

---

## 6. Run Automated Tests

### Step 1 — Check pytest

```powershell
python -m pytest --version
```

### Step 2 — Run all tests

```powershell
python -m pytest -v
```

### Step 3 — Run API tests only

```powershell
python -m pytest tests/test_api.py -v
```

### Step 4 — Run a specific test file

```powershell
python -m pytest tests/test_preprocessing.py -v
```

**Goal:** All expected tests pass before committing changes or building a new Docker image.

If a test fails, read the complete error message before changing code.

---

## 7. Git and GitHub Workflow

### Check the repository status

```powershell
git status
```

### Review changed files

```powershell
git diff
```

### Stage and commit changes

```powershell
git add .
git status
git commit -m "Describe the changes"
```

Replace `"Describe the changes"` with a meaningful message, such as:

```powershell
git commit -m "Fix FastAPI prediction endpoint"
```

### Push to GitHub

```powershell
git push origin main
```

This assumes the repository already has a remote named `origin` and the current branch is `main`.

### First-time setup only

Use these only if the repository has not already been initialized:

```powershell
git init
git branch -M main
git remote -v
```

If `origin` is not configured, add your actual repository URL:

```powershell
git remote add origin YOUR_GITHUB_REPOSITORY_URL
```

Do not run `git init` repeatedly or add a second remote if one already exists.

---

## 8. MLflow Experiment Tracking

### Start the MLflow UI

```powershell
python -m mlflow ui
```

Open:

`http://127.0.0.1:5000`

Stop the UI with `Ctrl + C`.

### What to check

- Experiment name: `ModelGuard-Income-Classification`
- Model parameters
- Accuracy and weighted F1-score
- Recall for the `>50K` class
- Quality-gate status
- Whether a model was accepted or rejected

The exact metrics depend on the latest training run. Always use the saved metrics and MLflow results rather than relying on old terminal output.

---

## 9. Docker Workflow

### Step 1 — Start Docker Desktop

If Docker Desktop is installed at the default location:

```powershell
Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
```

Wait for Docker Desktop to finish starting.

### Step 2 — Verify Docker

```powershell
docker --version
docker info
```

### Step 3 — Build the image

Run this from the project root containing the `Dockerfile`:

```powershell
docker build -t modelguard-api .
```

### Step 4 — Run the container

```powershell
docker run -d --name modelguard-api -p 8000:8000 modelguard-api
```

### Step 5 — Check running containers

```powershell
docker ps
```

Show all containers, including stopped ones:

```powershell
docker ps -a
```

### Step 6 — View container logs

```powershell
docker logs modelguard-api
```

Follow logs continuously:

```powershell
docker logs -f modelguard-api
```

### Step 7 — Test the containerized API

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### Step 8 — Stop the container

```powershell
docker stop modelguard-api
```

### Step 9 — Remove the container

```powershell
docker rm modelguard-api
```

If the container is still running, force-remove it:

```powershell
docker rm -f modelguard-api
```

### Step 10 — Check the image

```powershell
docker images modelguard-api
```

### Important Docker note

If a container named `modelguard-api` already exists, remove it before creating another container with the same name:

```powershell
docker rm -f modelguard-api
```

Then run the container command again.

This removes the container, not the Docker image. Do not remove a container if you need to preserve its current state.

---

## 10. `.dockerignore` Reference

The `.dockerignore` file prevents unnecessary files from being sent to Docker during image builds. This can reduce image size and build time.

Example:

```text
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.git/
.ipynb_checkpoints/
.env
.env.*
```

### Dataset and model files

Only ignore datasets and model files if the Docker build does not need them.

For example, if the API loads `models/modelguard_model.pkl` inside the container, **do not exclude that file** from the Docker build context.

Likewise, if the application needs a dataset or configuration file at runtime, ensure it is available inside the image or mounted correctly.

Never include credentials or API keys in the Docker image.

---

## 11. CI — Continuous Integration

**Purpose:** Automatically run checks when code is pushed to GitHub.

Recommended checks:

1. Install the required Python dependencies.
2. Run automated tests using pytest.
3. Fail the workflow if tests fail.
4. Add linting or Docker build validation later if useful.

Before implementing CI, confirm that the local test suite passes and the dependency file is up to date.

---

## 12. Quick Troubleshooting

| Problem | First thing to check |
|---|---|
| `python` uses the wrong environment | Activate `.venv` and check `python --version` |
| Module not found | Check the package installation and active environment |
| `uvicorn` cannot find `app` | Run from the project root and verify `app/main.py` |
| API returns HTTP 422 | Compare the JSON field names and data types with the request schema |
| API returns HTTP 500 | Read the complete traceback in the server terminal |
| Port 8000 is already in use | Stop the existing server or use another port |
| Docker daemon error | Start Docker Desktop and verify `docker info` |
| Docker container name conflict | Inspect `docker ps -a` and remove the old container if appropriate |
| Docker API behaves differently from local API | Check image contents, startup command, model path, and logs |
| Git push fails | Check `git status`, `git branch`, and `git remote -v` |

---

## 13. Recommended Daily Workflow

For ordinary development, follow this order:

1. Open the ModelGuard folder.
2. Activate `.venv`.
3. Start the FastAPI server if working on the API.
4. Make one small code change.
5. Run the relevant tests.
6. Inspect the output and fix any errors.
7. Commit the working change.
8. Push to GitHub when ready.
9. Rebuild and test Docker only when the containerized application needs updating.

**Rule:** Do not retrain the model, reinstall every package, recreate the environment, or rebuild Docker unnecessarily. First identify which part of the project you are changing.
