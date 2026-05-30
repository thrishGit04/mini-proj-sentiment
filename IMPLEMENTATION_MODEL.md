# AI Sentiment Analysis Tool: Model Implementation & Training Report

This document provides a comprehensive report of the technical stack, languages, tools, dataset, training methodology, data splitting, model architecture, evaluation metrics, and inference logic of the **AI Sentiment Analysis Tool**.

---

## 1. Executive Summary
The project is a full-stack, real-time sentiment analysis web application. It integrates a fine-tuned Deep Learning transformer model with a premium web-based interface. The model classifies user-provided text (specifically movie reviews, social media posts, or custom text) into **Positive**, **Negative**, or **Neutral** sentiments.

---

## 2. Technology Stack & Programming Languages

The application is structured into two main components: a web frontend and a machine learning backend API.

| Layer | Component | Languages & Technologies | Description |
| :--- | :--- | :--- | :--- |
| **Frontend** | User Interface | HTML5, CSS3, JavaScript (ES6+) | Single Page Application (SPA) designed with a "Cyber-Dark" glassmorphism theme. Statically served with interactive metrics charts, history storage, and real-time API integrations. |
| **Backend** | API Gateway | Python 3, FastAPI, Uvicorn | A lightweight ASGI REST server providing endpoints for sentiment prediction, health status checks, and session analytics. |
| **Machine Learning** | ML Engine | PyTorch (v2.x+), Hugging Face Transformers | Frameworks used for model building, fine-tuning, training pipelines, tokenization, and model inference. |

---

## 3. Libraries, Tools & Frameworks

### Backend & Machine Learning Tools
*   **PyTorch (`torch`)**: Used as the core deep learning framework to build and execute the custom classification network.
*   **Hugging Face `transformers`**:
    *   `DistilBertModel`: Used as the pre-trained transformer backbone.
    *   `DistilBertTokenizer`: Handles wordpiece tokenization, vocabulary mapping, padding, and truncation.
    *   `get_linear_schedule_with_warmup`: Sets up a learning rate scheduler that decays linearly after a warmup period.
*   **Scikit-Learn (`scikit-learn`)**:
    *   `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `roc_auc_score`, `classification_report`: Used to compute rigorous statistical evaluation metrics.
*   **Pandas (`pandas`)**: Used for tabular data loading, cleaning, mapping, and sampling from the Kaggle dataset.
*   **Kagglehub (`kagglehub`)**: Used to programmatically fetch the IMDb movie review dataset from Kaggle.
*   **FastAPI / Pydantic**: Provides robust HTTP REST API endpoints with automatic JSON request schema validation.
*   **Uvicorn**: An ASGI web server implementation for Python, running on port `8000`.

### Frontend Libraries
*   **Chart.js**: An open-source, HTML5-based charting library used to render dynamic charts (Doughnut and Bar charts) in the analytics dashboard.
*   **Google Fonts (Inter)**: Premium typography applied across all sections.

---

## 4. Dataset and Data Splitting

### Dataset Source
*   **Dataset Name**: IMDb Dataset of 50K Movie Reviews.
*   **Target Task**: Binary classification (sentiment labels: `positive` or `negative`).
*   **Pre-processing**: Label mapping was performed during data loading, converting `'positive' -> 1` and `'negative' -> 0`.

### Data Splitting Strategy
To ensure fast training runs and prevent long build times, a custom balanced subset of the 50,000 reviews was extracted:

1.  **Training Subset (5,000 samples)**:
    *   2,500 randomly sampled positive reviews.
    *   2,500 randomly sampled negative reviews.
    *   *Rationale*: Perfectly balanced (50% positive, 50% negative) to avoid class bias.
2.  **Validation/Testing Subset (1,000 samples)**:
    *   500 randomly sampled positive reviews (from the remaining pool).
    *   500 randomly sampled negative reviews (from the remaining pool).
    *   *Rationale*: Used as an independent test set for validation during training.
3.  **Data Loading**:
    *   PyTorch custom `Dataset` subclass (`IMDbDataset`) manages text encoding.
    *   PyTorch `DataLoader` wraps the datasets into batches:
        *   **Training Batch Size**: 32 (with shuffling enabled).
        *   **Validation Batch Size**: 64 (without shuffling).

---

## 5. Model Architecture & Transfer Learning

The model is custom-built, incorporating transfer learning techniques with a pre-trained transformer model.

```mermaid
graph TD
    A[Input Text] --> B[DistilBert Tokenizer]
    B --> C[Token IDs & Attention Masks]
    C --> D[DistilBERT Base Model]
    subgraph DistilBERT Backbone
        D1[Layer 1 - Frozen]
        D2[Layer 2 - Frozen]
        D3[Layer 3 - Frozen]
        D4[Layer 4 - Frozen]
        D5[Layer 5 - Fine-tuned]
        D6[Layer 6 - Fine-tuned]
        D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end
    D --> E[[CLS] Token Hidden State - Index 0]
    E --> F[Pre-Classifier Linear Layer 768 to 768]
    F --> G[ReLU Activation]
    G --> H[Dropout Layer p=0.3]
    H --> I[Classifier Linear Layer 768 to 2]
    I --> J[Logits Output]
```

### Transformer Backbone
*   **Pre-trained Model**: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions, 12 attention heads, ~66 million parameters).
*   **Layer Freezing Strategy (Fine-Tuning)**:
    *   The first **4 transformer layers** are frozen (`requires_grad = False`) to retain broad linguistic representations learned during pre-training.
    *   The last **2 transformer layers** are fine-tuned during training. This balances model specialization with computational efficiency.

### Custom Classification Head
A PyTorch custom module (`CustomDistilBertForClassification`) is appended directly to the output of DistilBERT:
1.  **Extraction**: The representation corresponding to the special `[CLS]` token (at index 0) is extracted from the last transformer layer's hidden states (`hidden_state[:, 0]`). This represents the aggregated sentence representation.
2.  **Pre-Classifier Layer**: A Linear layer projecting from 768 to 768 units.
3.  **Activation**: A rectified linear unit (`ReLU`) activation function.
4.  **Regularization**: A `Dropout` layer with a probability of $p = 0.3$ to prevent overfitting.
5.  **Classifier Layer**: A final Linear layer projecting from 768 units to 2 units (corresponding to class scores for Positive and Negative).

---

## 6. Training Configuration & Hyperparameters

During model training, the following optimization criteria and hyperparameters were configured:

*   **Maximum Sequence Length**: 256 tokens (longer reviews are truncated, shorter ones are padded with a `[PAD]` token).
*   **Epochs**: 3 (with Early Stopping patience of 1 epoch).
*   **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`).
*   **Optimizer**: `AdamW` (Adam with weight decay) using a learning rate of $2 \times 10^{-5}$ and default weight decay.
*   **Learning Rate Scheduler**: Linear decay schedule starting at $2 \times 10^{-5}$ and stepping down to $0.0$ at the end of training. Warmup steps were set to 0.
*   **Gradient Clipping**: Clipped at an L2 norm threshold of `1.0` to stabilize training gradients.
*   **Device Management**: CUDA (GPU) acceleration is automatically selected if available, falling back to CPU if absent.

---

## 7. Model Performance & Evaluation

The validation subset of 1,000 IMDb reviews yields the following performance metrics:

### Core Classification Metrics
*   **Accuracy**: `86.90%`
*   **Precision (Class 1 - Positive)**: `87.73%`
*   **Recall (Class 1 - Positive)**: `85.80%`
*   **F1-Score**: `86.75%`
*   **ROC-AUC Score**: `0.9417`

### Confusion Matrix
A total of 1,000 evaluations split evenly between Positive and Negative:

| Actual \ Predicted | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :---: | :---: |
| **Actual Negative (0)** | **440** (True Negative) | **60** (False Positive) |
| **Actual Positive (1)** | **71** (False Negative) | **429** (True Positive) |

### Detailed Classification Report

```json
{
    "Negative (Class 0)": {
        "precision": 0.861,
        "recall": 0.880,
        "f1-score": 0.870,
        "support": 500.0
    },
    "Positive (Class 1)": {
        "precision": 0.877,
        "recall": 0.858,
        "f1-score": 0.868,
        "support": 500.0
    },
    "Macro Average": {
        "precision": 0.869,
        "recall": 0.869,
        "f1-score": 0.869,
        "support": 1000.0
    }
}
```

---

## 8. Inference Pipeline & Neutral Class Synthesis

### Inference Engine
During prediction, the model runs inference using `torch.no_grad()` to avoid gradient computation and reduce memory usage. The input text is capped at `2000` characters to maintain quick response times.

### Synthesis of Neutral Sentiment
Because the IMDb dataset is annotated strictly for binary classification (positive or negative), the model was trained with $2$ output classes. To support a **Neutral** label in the live API and user interface, a mathematical synthesis is performed during post-processing:

1.  **Raw Probabilities**: Softmax is applied to the two logits to get raw probabilities for Positive ($P_{\text{pos\_raw}}$) and Negative ($P_{\text{neg\_raw}}$).
2.  **Neutral Synthesis**: The proximity of $P_{\text{pos\_raw}}$ and $P_{\text{neg\_raw}}$ is measured. If the confidence is split evenly, it indicates a neutral sentiment:
    $$P_{\text{neutral\_raw}} = 1.0 - |P_{\text{pos\_raw}} - P_{\text{neg\_raw}}|$$
3.  **Normalization**: The raw scores are normalized so that they sum to $1.0$:
    $$\text{Total} = P_{\text{pos\_raw}} + P_{\text{neg\_raw}} + P_{\text{neutral\_raw}}$$
    $$P_{\text{pos}} = \frac{P_{\text{pos\_raw}}}{\text{Total}}$$
    $$P_{\text{neg}} = \frac{P_{\text{neg\_raw}}}{\text{Total}}$$
    $$P_{\text{neutral}} = \frac{P_{\text{neutral\_raw}}}{\text{Total}}$$
4.  **Final Label Assignment**: The category with the highest normalized probability is returned as the final classification.

---

## 9. API Specifications (FastAPI backend)

The API is fully asynchronous and serves four routes:

*   `POST /analyze`:
    *   **Request JSON**: `{ "text": "The movie was alright..." }`
    *   **Response JSON**:
        ```json
        {
          "sentiment": "Neutral",
          "confidence": 0.354,
          "probabilities": {
            "positive": 0.323,
            "negative": 0.323,
            "neutral": 0.354
          },
          "inference_time_ms": 45,
          "text": "The movie was alright..."
        }
        ```
*   `GET /health`: Checks backend health and verifies whether `sentiment_model.pth` has loaded successfully.
*   `GET /metrics`: Returns aggregate statistics (total analyses, sentiment distributions) of the active server session.
*   `POST /clear-history`: Resets backend session metrics.

---

## 10. Project Directory Layout

```text
C:/Users/A2IN/Desktop/MINI PROP/
├── backend/
│   ├── app.py              # FastAPI REST Server
│   ├── train.py            # Training execution logic
│   ├── model.py            # Custom PyTorch model structure
│   ├── inference.py        # SentimentInference prediction class
│   ├── requirements.txt    # Python dependencies
│   └── saved_model/        # Output directory of model artifacts
│       ├── sentiment_model.pth
│       ├── metrics.json
│       └── tokenizer/
├── frontend/
│   ├── index.html          # Web UI layout
│   ├── style.css           # Premium cyber-dark theme styles
│   └── script.js           # Frontend state, API integration, and Chart.js setup
├── OVERVIEW_OF_PROJECT.md  # General project guide
├── SENTIMENT ANALYSIS TOOL - MINI PROJECT.md  # Specifications
└── IMPLEMENTATION_MODEL.md # This technical document
```
