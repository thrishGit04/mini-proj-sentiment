SENTIMENT ANALYSIS TOOL \- MINI PROJECT  
\`\`\`  
I need you to build a complete, production-ready Sentiment Analysis Tool with the following specifications:

\#\# PROJECT OVERVIEW  
Create a full-stack AI-powered sentiment analysis web application that:  
\- Uses the IMDb 50k movie review dataset from Kaggle  
\- Trains a transformer-based sentiment model (BERT or DistilBERT)  
\- Provides a visually stunning, cyber-dark themed web interface with glassmorphism effects  
\- Displays real-time sentiment predictions with confidence scores  
\- Includes analytics dashboards with charts and historical tracking

\#\# PART 1: MODEL TRAINING (Backend ML)

\#\#\# Dataset  
\- Use Kaggle's IMDb Dataset of 50,000 movie reviews (25k train, 25k test)  
\- Labels: Positive (1) and Negative (0) — binary classification  
\- If dataset download requires Kaggle API, provide instructions or use a local fallback

\#\#\# Model Architecture  
\- Use a pre-trained DistilBERT-base-uncased (lightning fast, near-BERT accuracy)  
\- Add a classification head with dropout (0.3) for regularization  
\- Freeze first 4 transformer layers, fine-tune last 2 layers

\#\#\# Training Configuration  
\- Maximum sequence length: 256 tokens  
\- Batch size: 32 for training, 64 for evaluation  
\- Epochs: 3 (with early stopping patience=1)  
\- Learning rate: 2e-5 with linear warmup  
\- Optimizer: AdamW  
\- Loss function: CrossEntropyLoss

\#\#\# Training Output  
After training, save:  
\- \`sentiment\_model.pth\` (PyTorch weights)  
\- \`tokenizer/\` folder with tokenizer files  
\- \`config.json\` with model parameters  
\- \`metrics.json\` containing train/validation accuracy, loss curves

\#\#\# Evaluation Metrics  
Generate and display:  
\- Accuracy, Precision, Recall, F1-Score  
\- Confusion Matrix  
\- ROC-AUC score  
\- Classification report

\#\#\# Expected Performance  
\- Target validation accuracy: 88%–92% on IMDb test set

\#\# PART 2: FRONTEND WEBSITE (Visual \+ Interactive)

\#\#\# Design Aesthetic (Strictly follow)  
Theme: \*\*"Cyber-Dark Glow"\*\* with Glassmorphism

\*\*Color Palette:\*\*  
\- Background: Deep navy/black (\#0A0F1F) with animated floating gradient orbs (Indigo \#4338CA and Cyan \#06B6D4)  
\- Glass cards: rgba(15, 25, 45, 0.65) with backdrop-blur(20px), border rgba(255,255,255,0.15), border-radius 24px  
\- Text: White (\#FFFFFF) for primary, Slate-300 (\#CBD5E1) for secondary  
\- Accent gradients: Cyan-to-Purple (\#06B6D4 to \#8B5CF6)  
\- Success: \#10B981 (Emerald), Danger: \#EF4444 (Red), Warning: \#F59E0B (Amber)

\*\*Typography:\*\*  
\- Font family: 'Inter', sans-serif (from Google Fonts)  
\- Headings: Semi-bold 600, Body: Regular 400

\#\#\# Layout Structure (Single Page Application)

\#\#\#\# Section 1: Hero Header  
\- Animated SVG logo with pulsing glow effect  
\- Title: "Sentiment Analysis Tool — AI Powered"  
\- Subtitle: "Powered by DistilBERT | Trained on 50K IMDb Reviews"  
\- Live status indicator: "Model Ready • Real-time Analysis"

\#\#\#\# Section 2: Input Workspace  
\- Large textarea (placeholder: "Write a movie review or paste any text...")  
\- Character counter: x/2000 characters (dynamic, turns red near limit)  
\- Three "Quick Sample" chips/pills:  
  \- "⭐ Best movie ever\! Absolutely loved it."  
  \- "💩 Terrible acting, waste of time."  
  \- "😐 It was okay, nothing special."  
\- Two buttons:  
  \- "Analyze Sentiment" — gradient filled, large, with hover animation  
  \- "Clear" — outlined subtle button  
\- Shake animation on empty submission

\#\#\#\# Section 3: Real-Time Result Card (animated reveal)  
\- Large sentiment badge: POSITIVE (😊) / NEGATIVE (😞) / NEUTRAL (😐)  
\- Confidence percentage (e.g., "97.3% confident")  
\- Three live progress bars with labels:  
  \- Positive \[Green\] — fill width \= probability %  
  \- Negative \[Red\] — fill width \= probability %  
  \- Neutral \[Amber\] — fill width \= probability %  
\- Sparkline showing probability distribution

\#\#\#\# Section 4: Analytics Dashboard (visible after 2+ analyses)  
\- \*\*Doughnut Chart:\*\* Overall sentiment distribution (Positive vs Negative vs Neutral)  
\- \*\*Bar Chart:\*\* Confidence comparison across last 10 analyses  
\- \*\*Line Chart (optional):\*\* Sentiment polarity over time  
\- All charts using Chart.js or ApexCharts, with dark theme integration

\#\#\#\# Section 5: History Vault  
\- Scrollable table with columns:  
  | \# | Text (truncated) | Sentiment | Confidence | Timestamp |  
\- Each row has color-coded sentiment pill  
\- "Clear History" button  
\- Maximum 50 stored entries (auto oldest removal)

\#\#\#\# Footer  
\- Model accuracy: \[X\]% | Framework: PyTorch \+ Transformers | Dataset: IMDb 50k  
\- GitHub/project reference (placeholder)

\#\#\# Micro-interactions (Required)  
1\. \*\*Typing animation\*\* on result reveal  
2\. \*\*Button loading state\*\* — spinner appears on Analyze click, button text changes to "Analyzing..."  
3\. \*\*Smooth auto-scroll\*\* to results after analysis  
4\. \*\*Hover lift effect\*\* on cards (scale: 1.02, transition: 0.2s)  
5\. \*\*Glassmorphism border pop\*\* on focus for textarea

\#\#\# Responsive Behavior  
\- Desktop (≥1024px): Two-column layout for charts (side by side)  
\- Tablet (768px–1023px): Single column, reduced padding  
\- Mobile (\<768px): Stacked layout, font-size reduced, buttons full width, chart containers scroll horizontally if needed

\#\# PART 3: INTEGRATION & WORKFLOW

\#\#\# Backend API Requirements (FastAPI or Flask)  
Create REST endpoints:

| Endpoint | Method | Description |  
|----------|--------|-------------|  
| \`/analyze\` | POST | Accepts JSON { "text": "user input" } → returns { "sentiment": "POSITIVE", "confidence": 0.94, "probabilities": {"positive": 0.94, "negative": 0.05, "neutral": 0.01} } |  
| \`/health\` | GET | Returns model status |  
| \`/metrics\` | GET | Returns global session stats (total analyses, positive/negative ratio) |

\#\#\# Model Inference Pipeline  
1\. Tokenize input text (same tokenizer used in training)  
2\. Pad/truncate to 256 tokens  
3\. Run model inference (disable gradient to save memory)  
4\. Apply softmax to get probabilities  
5\. Map highest probability to sentiment label  
6\. Return JSON response (under 100ms target)

\#\#\# Session & State  
\- History stored in browser localStorage (not backend)  
\- Stats aggregated from localStorage history  
\- No authentication required (public demo)

\#\# PART 4: DEPLOYMENT & DELIVERABLES

\#\#\# File Structure (to be generated)  
\`\`\`  
sentiment-tool/  
├── backend/  
│   ├── train.py (training script)  
│   ├── app.py (API server)  
│   ├── model.py (model definition)  
│   ├── inference.py (prediction logic)  
│   ├── requirements.txt  
│   └── saved\_model/  
├── frontend/  
│   ├── index.html (complete single file)  
│   ├── style.css (or embedded in HTML)  
│   ├── script.js (or embedded)  
│   └── assets/ (logos, favicon)  
└── README.md (setup \+ run instructions)  
\`\`\`

\#\#\# Running Instructions (must work)  
Provide clear commands:  
\`\`\`bash  
\# Training  
python backend/train.py \--epochs 3 \--batch\_size 32

\# Start API  
uvicorn backend.app:app \--reload \--port 8000

\# Frontend (just open index.html or serve with live-server)  
\`\`\`

\#\#\# Optional but Recommended  
\- Use ONNX runtime or quantization for faster inference  
\- Cache recent predictions in memory (LRU cache with 100 entries)  
\- Show estimated inference time in UI (e.g., "Analyzed in 42ms")

\#\# PART 5: QUALITY REQUIREMENTS

1\. \*\*No placeholder AI\*\* — The model MUST actually train on IMDb data, not use a generic sentiment API  
2\. \*\*Fully self-contained\*\* — All code must be provided; no external paid APIs  
3\. \*\*Error handling\*\* — Gracefully handle empty text, very long text, special characters, emojis  
4\. \*\*Mobile-first CSS\*\* — Use clamp(), flexbox/grid, media queries  
5\. \*\*Cross-browser\*\* — Works on Chrome, Firefox, Safari latest versions  
6\. \*\*Performance\*\* — Initial load \< 2s, inference \< 200ms on CPU

\#\# EXAMPLE API REQUEST/RESPONSE

\*\*Request:\*\*  
\`\`\`json  
{  
  "text": "This movie was absolutely fantastic\! The acting was superb."  
}  
\`\`\`

\*\*Response:\*\*  
\`\`\`json  
{  
  "sentiment": "POSITIVE",  
  "confidence": 0.967,  
  "probabilities": {  
    "positive": 0.967,  
    "negative": 0.021,  
    "neutral": 0.012  
  },  
  "inference\_time\_ms": 48  
}  
\`\`\`

