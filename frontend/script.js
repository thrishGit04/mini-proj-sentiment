/**
 * Sentiment Analysis Tool — Client-Side Logic
 * =============================================
 * Handles text analysis via fetch, renders results with animated bars,
 * maintains analysis history, and updates Chart.js pie & bar charts.
 */

// ── DOM Elements ────────────────────────────────────────────────────────────
const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const resultSection = document.getElementById("result-section");
const resultBadge = document.getElementById("result-badge");
const resultEmoji = document.getElementById("result-emoji");
const resultLabel = document.getElementById("result-label");
const confidenceValue = document.getElementById("confidence-value");
const resultTextDisp = document.getElementById("result-text-display");
const chartsSection = document.getElementById("charts-section");
const historySection = document.getElementById("history-section");
const historyBody = document.getElementById("history-body");
const clearHistoryBtn = document.getElementById("clear-history-btn");

// ── State ───────────────────────────────────────────────────────────────────
let history = [];
let pieChart = null;
let barChart = null;

// ── API Configuration ───────────────────────────────────────────────────────
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://sentiment-analysis-backend-47jb.onrender.com';


// ── Emoji / class maps ──────────────────────────────────────────────────────
const emojiMap = { Positive: "😊", Negative: "😠", Neutral: "😐" };
const classMap = { Positive: "positive", Negative: "negative", Neutral: "neutral" };
const pillMap = { Positive: "pill-positive", Negative: "pill-negative", Neutral: "pill-neutral" };

// ── Character counter ───────────────────────────────────────────────────────
textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} / 2000`;
});

// ── Clear input ─────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
    textInput.value = "";
    charCount.textContent = "0 / 2000";
    textInput.focus();
});

// ── Sample chips ────────────────────────────────────────────────────────────
document.querySelectorAll(".sample-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        textInput.value = chip.dataset.text;
        charCount.textContent = `${textInput.value.length} / 2000`;
        textInput.focus();
    });
});

// ── Analyze button ──────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

// Allow Ctrl+Enter to submit
textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        runAnalysis();
    }
});

async function runAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
        shakeElement(textInput);
        return;
    }

    // Show loading state
    setLoading(true);

    try {
        const res = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Analysis failed");
        }

        const data = await res.json();
        history.push(data);
        displayResult(data);
        updateCharts();
        updateHistory();

    } catch (err) {
        console.error("Analysis error:", err);
        alert("Error: " + err.message);
    } finally {
        setLoading(false);
    }
}

function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.querySelector(".btn-text").style.display = loading ? "none" : "inline";
    analyzeBtn.querySelector(".btn-loader").style.display = loading ? "inline-flex" : "none";
}

function shakeElement(el) {
    el.style.animation = "none";
    el.offsetHeight; // reflow
    el.style.animation = "shake 0.4s ease";
    setTimeout(() => el.style.animation = "", 400);
}

// ── Display Result ──────────────────────────────────────────────────────────
function displayResult(data) {
    resultSection.classList.remove("hidden");

    // Badge
    resultBadge.className = `result-badge ${classMap[data.sentiment]}`;
    resultEmoji.textContent = emojiMap[data.sentiment] || "🤔";
    resultLabel.textContent = data.sentiment;

    // Confidence
    confidenceValue.textContent = (data.confidence * 100).toFixed(1) + "%";

    // Breakdown bars (animate)
    const scores = data.probabilities || {};
    setTimeout(() => {
        animateBar("bar-positive", "pct-positive", scores.positive || 0);
        animateBar("bar-neutral", "pct-neutral", scores.neutral || 0);
        animateBar("bar-negative", "pct-negative", scores.negative || 0);
    }, 100);

    // Text preview
    resultTextDisp.textContent = data.text.length > 300
        ? data.text.substring(0, 300) + "..."
        : data.text;

    // Scroll into view
    resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
}

function animateBar(barId, pctId, score) {
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    const pctVal = (score * 100).toFixed(1);
    bar.style.width = pctVal + "%";
    pct.textContent = pctVal + "%";
}

// ── Update Charts ───────────────────────────────────────────────────────────
function updateCharts() {
    if (history.length === 0) {
        chartsSection.classList.add("hidden");
        return;
    }

    chartsSection.classList.remove("hidden");

    // Count sentiments
    const counts = { Positive: 0, Negative: 0, Neutral: 0 };
    history.forEach(h => counts[h.sentiment] = (counts[h.sentiment] || 0) + 1);

    const chartColors = {
        bg: ["rgba(52,211,153,0.7)", "rgba(251,191,36,0.7)", "rgba(248,113,113,0.7)"],
        border: ["#34d399", "#fbbf24", "#f87171"],
    };

    // ── Pie Chart ──
    const pieCtx = document.getElementById("pie-chart").getContext("2d");
    if (pieChart) pieChart.destroy();
    pieChart = new Chart(pieCtx, {
        type: "doughnut",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [counts.Positive, counts.Neutral, counts.Negative],
                backgroundColor: chartColors.bg,
                borderColor: chartColors.border,
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: "55%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#94a3b8", font: { family: "'Inter', sans-serif", size: 12 }, padding: 16 },
                },
            },
        },
    });

    // ── Bar Chart (last 10 entries confidence comparison) ──
    const barCtx = document.getElementById("bar-chart").getContext("2d");
    if (barChart) barChart.destroy();

    const recent = history.slice(-10);
    const barLabels = recent.map((_, i) => `#${history.length - recent.length + i + 1}`);
    const barData = recent.map(h => (h.confidence * 100).toFixed(1));
    const barColors = recent.map(h => {
        if (h.sentiment === "Positive") return "rgba(52,211,153,0.7)";
        if (h.sentiment === "Negative") return "rgba(248,113,113,0.7)";
        return "rgba(251,191,36,0.7)";
    });
    const barBorders = recent.map(h => {
        if (h.sentiment === "Positive") return "#34d399";
        if (h.sentiment === "Negative") return "#f87171";
        return "#fbbf24";
    });

    barChart = new Chart(barCtx, {
        type: "bar",
        data: {
            labels: barLabels,
            datasets: [{
                label: "Confidence %",
                data: barData,
                backgroundColor: barColors,
                borderColor: barBorders,
                borderWidth: 2,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: "#64748b", font: { size: 11 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
                x: {
                    ticks: { color: "#64748b", font: { size: 11 } },
                    grid: { display: false },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}

// ── Update History Table ────────────────────────────────────────────────────
function updateHistory() {
    if (history.length === 0) {
        historySection.classList.add("hidden");
        return;
    }

    historySection.classList.remove("hidden");
    historyBody.innerHTML = "";

    // Show newest first
    [...history].reverse().forEach((item, idx) => {
        const row = document.createElement("tr");
        const truncatedText = item.text.length > 60
            ? item.text.substring(0, 60) + "…"
            : item.text;

        row.innerHTML = `
            <td>${history.length - idx}</td>
            <td class="table-text-cell" title="${escapeHtml(item.text)}">${escapeHtml(truncatedText)}</td>
            <td><span class="sentiment-pill ${pillMap[item.sentiment]}">${emojiMap[item.sentiment]} ${item.sentiment}</span></td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
            <td>${item.timestamp}</td>
        `;
        row.style.animation = idx === 0 ? "fadeInUp 0.3s ease-out" : "none";
        historyBody.appendChild(row);
    });
}

// ── Clear History ───────────────────────────────────────────────────────────
clearHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Clear all analysis history?")) return;

    try {
        await fetch(`${API_BASE_URL}/clear-history`, { method: "POST" });
        history = [];
        historyBody.innerHTML = "";
        historySection.classList.add("hidden");
        chartsSection.classList.add("hidden");
        resultSection.classList.add("hidden");
        if (pieChart) { pieChart.destroy(); pieChart = null; }
        if (barChart) { barChart.destroy(); barChart = null; }
    } catch (err) {
        console.error("Failed to clear history:", err);
    }
});

// ── Utility ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Add shake keyframe dynamically ──────────────────────────────────────────
const style = document.createElement("style");
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-6px); }
        40% { transform: translateX(6px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
`;
document.head.appendChild(style);
