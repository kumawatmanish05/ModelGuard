
const API_BASE = ""; // Same origin: FastAPI will serve the frontend.

// Load model metrics from the API
async function loadModelMetrics() {
    const metricsSection = document.getElementById("metrics");

    try {
        const response = await fetch(`${API_BASE}/model-info`);

        if (!response.ok) {
            throw new Error("Could not load model metrics");
        }

        const metrics = await response.json();

        metricsSection.innerHTML = `
            <h2>Model Metrics</h2>
            <p><strong>Selected model:</strong> ${metrics.selected_model ?? "N/A"}</p>
            <p><strong>Accuracy:</strong> ${formatMetric(metrics.accuracy)}</p>
            <p><strong>Weighted F1:</strong> ${formatMetric(metrics.f1_weighted)}</p>
            <p><strong>Recall (>50K):</strong> ${formatMetric(metrics.recall_over_50k)}</p>
        `;
    } catch (error) {
        metricsSection.innerHTML = `
            <h2>Model Metrics</h2>
            <p>Unable to load metrics. Start the ModelGuard API and refresh.</p>
        `;
        console.error(error);
    }
}

// Load the model quality-gate status
async function loadQualityStatus() {
    const qualitySection = document.getElementById("quality");

    try {
        const response = await fetch(`${API_BASE}/quality-check`);

        if (!response.ok) {
            throw new Error("Could not load quality status");
        }

        const quality = await response.json();

        qualitySection.innerHTML = `
            <h2>Quality Gate</h2>
            <p><strong>Status:</strong> ${
                quality.quality_gate_passed ? "PASS" : "FAIL"
            }</p>
            <p><strong>Recall achieved:</strong> ${formatMetric(quality.recall_over_50k)}</p>
            <p><strong>Required recall:</strong> ${formatMetric(quality.required_recall)}</p>
        `;
    } catch (error) {
        qualitySection.innerHTML = `
            <h2>Quality Gate</h2>
            <p>Unable to load quality status. Check the API connection.</p>
        `;
        console.error(error);
    }
}

// Format decimal metrics as percentages
function formatMetric(value) {
    const number = Number(value);

    if (value === null || value === undefined || !Number.isFinite(number)) {
        return "N/A";
    }

    return `${(number * 100).toFixed(2)}%`;
}

// Load dashboard data when the page opens
document.addEventListener("DOMContentLoaded", () => {
    loadModelMetrics();
    loadQualityStatus();
});



const predictionForm = document.getElementById("prediction-form");

if (predictionForm) {
    predictionForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const resultBox = document.getElementById("prediction-result");
        const submitButton = predictionForm.querySelector('button[type="submit"]');

        const formData = new FormData(predictionForm);
        const applicant = Object.fromEntries(formData.entries());

        // Convert numeric fields from strings to numbers
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
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(applicant)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Prediction request failed");
            }

            const isHighIncome = data.prediction.trim() === ">50K";

            resultBox.innerHTML = `
                <h3>Prediction Result</h3>
                <p>
                    <strong>Predicted income:</strong>
                    ${data.prediction}
                </p>
                <p>
                    ${isHighIncome
                        ? "The model predicts income above $50K."
                        : "The model predicts income of $50K or below."}
                </p>
                <small>This is a model prediction, not a guarantee of actual income.</small>
            `;
        } catch (error) {
            resultBox.textContent = `Prediction failed: ${error.message}`;
            console.error("Prediction error:", error);
        } finally {
            submitButton.disabled = false;
        }
    });
}


