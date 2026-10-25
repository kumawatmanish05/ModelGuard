
const API_BASE = ""; // Same origin: FastAPI serves the frontend and API.

// Format decimal metrics as percentages
function formatMetric(value) {
    const number = Number(value);

    if (
        value === null ||
        value === undefined ||
        value === "" ||
        !Number.isFinite(number)
    ) {
        return "N/A";
    }

    return `${(number * 100).toFixed(2)}%`;
}

// Load model metrics from the API
async function loadModelMetrics() {
    const metricsSection = document.getElementById("metrics");

    if (!metricsSection) return;

    try {
        const response = await fetch(`${API_BASE}/model-info`);

        if (!response.ok) {
            throw new Error("Could not load model metrics");
        }

        const metrics = await response.json();

        metricsSection.innerHTML = `
            <article class="metric-card">
                <span class="metric-label">Selected model</span>
                <h3>${metrics.selected_model ?? "N/A"}</h3>
                <p>Active classifier</p>
            </article>

            <article class="metric-card">
                <span class="metric-label">Accuracy</span>
                <h3>${formatMetric(metrics.accuracy)}</h3>
                <p>Overall correctness</p>
            </article>

            <article class="metric-card">
                <span class="metric-label">Weighted F1</span>
                <h3>${formatMetric(metrics.f1_weighted)}</h3>
                <p>Precision and recall balance</p>
            </article>

            <article class="metric-card accent-card">
                <span class="metric-label">Recall · &gt;50K</span>
                <h3>${formatMetric(metrics.recall_over_50k)}</h3>
                <p>Positive-class detection</p>
            </article>
        `;
    } catch (error) {
        metricsSection.innerHTML = `
            <article class="metric-card">
                <span class="metric-label">Model metrics</span>
                <h3>Unavailable</h3>
                <p>Unable to load model metrics. Check the API and refresh.</p>
            </article>
        `;

        console.error("Metrics loading error:", error);
    }
}

// Load the model quality-gate status
async function loadQualityStatus() {
    const qualitySection = document.getElementById("quality");

    if (!qualitySection) return;

    try {
        const response = await fetch(`${API_BASE}/quality-check`);

        if (!response.ok) {
            throw new Error("Could not load quality status");
        }

        const quality = await response.json();
        const passed = quality.quality_gate_passed;

        qualitySection.innerHTML = `
            <div class="quality-icon" aria-hidden="true">
                ${passed ? "✓" : "!"}
            </div>

            <div class="quality-copy">
                <p class="eyebrow">MODEL GOVERNANCE</p>
                <h2>Quality gate</h2>
                <p>
                    Recall achieved:
                    ${formatMetric(quality.recall_over_50k)}
                    · Required:
                    ${formatMetric(quality.required_recall)}
                </p>
            </div>

            <div class="quality-result"
                 style="${
                     passed
                         ? ""
                         : "background:#fff0f0;color:#b42318;"
                 }">
                <span class="status-dot"
                      style="${
                          passed
                              ? ""
                              : "background:#dc2626;box-shadow:0 0 0 4px #dc26261a;"
                      }"></span>
                <span>${passed ? "PASS" : "FAIL"}</span>
            </div>
        `;
    } catch (error) {
        qualitySection.innerHTML = `
            <div class="quality-icon" aria-hidden="true">!</div>

            <div class="quality-copy">
                <p class="eyebrow">MODEL GOVERNANCE</p>
                <h2>Quality gate</h2>
                <p>
                    Unable to load quality status.
                    Check the API connection and refresh.
                </p>
            </div>

            <div class="quality-result"
                 style="background:#fff0f0;color:#b42318;">
                Unavailable
            </div>
        `;

        console.error("Quality-gate loading error:", error);
    }
}

// Handle income prediction form
function setupPredictionForm() {
    const predictionForm = document.getElementById("prediction-form");

    if (!predictionForm) return;

    predictionForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const resultBox = document.getElementById("prediction-result");
        const submitButton = predictionForm.querySelector(
            'button[type="submit"]'
        );

        const formData = new FormData(predictionForm);
        const applicant = Object.fromEntries(formData.entries());

        // Convert numeric form fields from strings to numbers
        const numericFields = [
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week"
        ];

        numericFields.forEach((field) => {
            applicant[field] = Number(applicant[field]);
        });

        resultBox.textContent = "Predicting income...";
        submitButton.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(applicant)
            });

            const data = await response.json();

            if (!response.ok) {
                const detail = Array.isArray(data.detail)
                    ? data.detail
                        .map((item) => item.msg || "Invalid input")
                        .join(", ")
                    : data.detail;

                throw new Error(
                    detail || "Prediction request failed"
                );
            }

            if (typeof data.prediction !== "string") {
                throw new Error("The API returned an unexpected response");
            }

            const prediction = data.prediction.trim();
            const isHighIncome = prediction === ">50K";

            resultBox.innerHTML = `
                <h3>Prediction result</h3>
                <p>
                    <strong>Predicted income:</strong>
                    <span>${prediction}</span>
                </p>
                <p>
                    ${
                        isHighIncome
                            ? "The model predicts income above $50K."
                            : "The model predicts income of $50K or below."
                    }
                </p>
                <small>
                    This is a model prediction, not a guarantee
                    of actual income.
                </small>
            `;

            resultBox.style.borderStyle = "solid";
            resultBox.style.borderColor = isHighIncome
                ? "#bde9dc"
                : "#dfe2ec";
            resultBox.style.background = isHighIncome
                ? "#f0fbf7"
                : "#fafaff";
        } catch (error) {
            resultBox.textContent = `Prediction failed: ${error.message}`;
            resultBox.style.borderColor = "#f0b8b8";
            resultBox.style.background = "#fff5f5";

            console.error("Prediction error:", error);
        } finally {
            submitButton.disabled = false;
        }
    });
}

// Initialize dashboard after the HTML has loaded
document.addEventListener("DOMContentLoaded", () => {
    loadModelMetrics();
    loadQualityStatus();
    setupPredictionForm();
});



