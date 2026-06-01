# Sentiment Analysis Tool — AI-Powered Text Classification

A premium, full-stack sentiment analysis application designed with a cyber-dark theme and glassmorphism UI. It uses a custom-trained PyTorch transformer model based on DistilBERT fine-tuned on the IMDb Movie Reviews dataset.

## Live Demo
*Link to your Render URL after deployment: [https://sentiment-analysis-frontend-0q5l.onrender.com](https://sentiment-analysis-frontend-0q5l.onrender.com)*

---

## 🚀 Key Features

*   **Interactive Analysis**: Real-time sentiment classification into Positive, Neutral, or Negative categories.
*   **Confidence Breakdown**: Animated progress bars showing exact category probability distributions.
*   **Analytics Dashboard**: Interactive charts (Doughnut & Bar charts) powered by Chart.js tracking mood mix and historical confidence values.
*   **Analysis History Vault**: Local session tracking with timestamps, allowing session logs to be saved and cleared.
*   **Robust Backend**: Powered by FastAPI with an automatic mock fallback if model weights are not loaded.

---

## 🛠️ Tech Stack & Architecture

*   **Frontend**: HTML5, Vanilla CSS3 (Custom Cyber-Dark design), JavaScript (ES6+), Chart.js
*   **Backend**: Python 3, FastAPI, Uvicorn, Pydantic
*   **Machine Learning**: PyTorch, Hugging Face Transformers (`distilbert-base-uncased`), Scikit-Learn

---

## 📂 Project Directory Structure

```text
├── backend/
│   ├── app.py              # FastAPI REST Server
│   ├── train.py            # Training execution logic
│   ├── model.py            # Custom PyTorch model structure
│   ├── inference.py        # SentimentInference prediction class
│   ├── requirements.txt    # Python dependencies
│   └── saved_model/        # Model artifacts (gitignored due to size)
├── frontend/
│   ├── index.html          # Web UI layout
│   ├── style.css           # Custom glassmorphic styling
│   └── script.js           # API integration, State management, Chart.js
├── render.yaml             # Render Blueprint configuration
├── OVERVIEW_OF_PROJECT.md  # Detailed UX description
├── IMPLEMENTATION_MODEL.md # Deep technical architecture report
└── README.md               # Setup & Run guide
```

---

## 💻 Local Setup & Execution

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Install Dependencies
Navigate to the `backend` folder and install the dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Model Training
To train the model on your local machine:
```bash
python train.py --epochs 3 --batch_size 32
```
*Note: This will download the IMDb dataset via `kagglehub`, train the custom classification head for 3 epochs, and save the best model weights to `backend/saved_model/`.*

### 3. Run the Backend API Server
Start the Uvicorn ASGI server:
```bash
uvicorn app:app --reload --port 8000
```
The backend API documentation will be available at `http://127.0.0.1:8000/docs`.

### 4. Run the Frontend UI
Simply open `frontend/index.html` in any web browser, or serve it using a local server (e.g., Python's HTTP server):
```bash
cd ../frontend
python -m http.server 3000
```
Open `http://localhost:3000` in your web browser.

---

## ☁️ Deployment on Render

This project is configured as a multi-service blueprint in `render.yaml` for seamless deployment on **Render.com**.

### Automated Blueprint Deployment
1.  Push the project code to a public or private GitHub repository.
2.  Log in to [Render](https://render.com).
3.  Click **New +** -> **Blueprint**.
4.  Connect your GitHub repository. Render will automatically detect `render.yaml` and configure both the backend (FastAPI Web Service) and the frontend (Static Site).
5.  Click **Apply**.

---

## ⚠️ Notes for Free-Tier Hosting

*   **Inference Fallback**: The model weights (`saved_model/sentiment_model.pth`) are larger than 100MB and are excluded from Git using `.gitignore`. On Render free tier, the API will load a mock prediction fallback to allow testing the UI.
*   **Cold Starts**: The backend service spins down after 15 minutes of inactivity on Render's free tier. The first request may take 30-50 seconds to complete while the container wakes up.
*   **Production Weights Deployment**: To deploy the actual trained model, host the `sentiment_model.pth` file on an external storage provider (e.g., Hugging Face Spaces LFS, AWS S3, or Google Cloud Storage) and update `backend/app.py` to download it programmatically on startup.
