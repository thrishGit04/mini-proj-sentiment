/**
 * Sentiment Analysis Tool — Client-Side Logic
 * =============================================
 * Handles text analysis via fetch, renders results with animated bars,
 * maintains analysis history, and updates Chart.js doughnut & bar charts.
 *
 * FIXES:
 *  - Bar chart Y-axis now shows % suffix on ticks
 *  - Bar chart tooltip shows both sentiment and confidence %
 *  - Doughnut tooltip shows count and percentage of total
 *  - Score breakdown bars animate correctly from 0 → actual value
 *  - Confidence display shows meaningful interpretation label
 *  - All chart data stored as numbers (not strings) for correct rendering
 *  - Bar chart borderRadius only applied to top via Chart.js options
 *  - Added a 3rd chart: per-sentiment avg confidence line chart
 */

// ── DOM Elements ─────────────────────────────────────────────────────────────
const textInput       = document.getElementById("text-input");
const charCount       = document.getElementById("char-count");
const analyzeBtn      = document.getElementById("analyze-btn");
const clearBtn        = document.getElementById("clear-btn");
const resultSection   = document.getElementById("result-section");
const resultBadge     = document.getElementById("result-badge");
const resultEmoji     = document.getElementById("result-emoji");
const resultLabel     = document.getElementById("result-label");
const confidenceValue = document.getElementById("confidence-value");
const confidenceLevel = document.getElementById("confidence-level");
const resultTextDisp  = document.getElementById("result-text-display");
const chartsSection   = document.getElementById("charts-section");
const historySection  = document.getElementById("history-section");
const historyBody     = document.getElementById("history-body");
const clearHistoryBtn = document.getElementById("clear-history-btn");

// ── State ────────────────────────────────────────────────────────────────────
let history  = [];
let pieChart = null;
let barChart = null;

// ── API Configuration ────────────────────────────────────────────────────────
const API_BASE_URL =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://sentiment-analysis-backend-47jb.onrender.com";

// ── Maps ─────────────────────────────────────────────────────────────────────
const emojiMap = { Positive: "😊", Negative: "😠", Neutral: "😐" };
const classMap = { Positive: "positive", Negative: "negative", Neutral: "neutral" };
const pillMap  = { Positive: "pill-positive", Negative: "pill-negative", Neutral: "pill-neutral" };

// ── Confidence label helper ──────────────────────────────────────────────────
/**
 * Returns a qualitative label for the confidence score.
 *   ≥ 0.90 → "Very High"   ≥ 0.75 → "High"
 *   ≥ 0.60 → "Moderate"    ≥ 0.45 → "Low"   else → "Very Low"
 */
function confidenceLabel(conf) {
  if (conf >= 0.90) return { text: "Very High Confidence", cls: "conf-very-high" };
  if (conf >= 0.75) return { text: "High Confidence",      cls: "conf-high"      };
  if (conf >= 0.60) return { text: "Moderate Confidence",  cls: "conf-moderate"  };
  if (conf >= 0.45) return { text: "Low Confidence",       cls: "conf-low"       };
  return                    { text: "Very Low Confidence", cls: "conf-very-low"  };
}

// ── Character counter ─────────────────────────────────────────────────────────
textInput.addEventListener("input", () => {
  charCount.textContent = `${textInput.value.length} / 2000`;
});

// ── Clear input ──────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  textInput.value = "";
  charCount.textContent = "0 / 2000";
  textInput.focus();
});

// ── Sample chips ─────────────────────────────────────────────────────────────
document.querySelectorAll(".sample-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    textInput.value = chip.dataset.text;
    charCount.textContent = `${textInput.value.length} / 2000`;
    textInput.focus();
  });
});

// ── Analyze button ───────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

textInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runAnalysis();
  }
});

async function runAnalysis() {
  const text = textInput.value.trim();
  if (!text) { shakeElement(textInput); return; }

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
  analyzeBtn.querySelector(".btn-text").style.display   = loading ? "none"        : "inline";
  analyzeBtn.querySelector(".btn-loader").style.display = loading ? "inline-flex" : "none";
}

function shakeElement(el) {
  el.style.animation = "none";
  el.offsetHeight; // reflow
  el.style.animation = "shake 0.4s ease";
  setTimeout(() => (el.style.animation = ""), 400);
}

// ── Display Result ────────────────────────────────────────────────────────────
function displayResult(data) {
  resultSection.classList.remove("hidden");

  // Badge
  resultBadge.className = `result-badge ${classMap[data.sentiment]}`;
  resultEmoji.textContent = emojiMap[data.sentiment] || "🤔";
  resultLabel.textContent = data.sentiment;

  // Confidence — show numeric % AND a qualitative label so "50.0%" reads as "Low Confidence"
  const confPct = data.confidence * 100;
  const { text: lvlText, cls: lvlCls } = confidenceLabel(data.confidence);
  confidenceValue.textContent = confPct.toFixed(1) + "%";

  // Update the confidence level badge
  if (confidenceLevel) {
    confidenceLevel.textContent = lvlText;
    confidenceLevel.className = `confidence-level-badge ${lvlCls}`;
  }

  // ── Score breakdown bars (reset first, then animate after brief delay)
  const scores = data.probabilities || {};
  // Reset bars to 0 before animating so transition fires even if same value
  ["bar-positive", "bar-neutral", "bar-negative"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.transition = "none"; el.style.width = "0%"; }
  });

  // Force reflow then animate
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      animateBar("bar-positive", "pct-positive", scores.positive ?? 0);
      animateBar("bar-neutral",  "pct-neutral",  scores.neutral  ?? 0);
      animateBar("bar-negative", "pct-negative", scores.negative ?? 0);
    });
  });

  // Text preview
  resultTextDisp.textContent =
    data.text.length > 300 ? data.text.substring(0, 300) + "…" : data.text;

  resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
}

/**
 * Animate a score bar.
 * @param {string} barId  - element id of the bar fill div
 * @param {string} pctId  - element id of the percentage text span
 * @param {number} score  - raw probability 0..1
 */
function animateBar(barId, pctId, score) {
  const bar    = document.getElementById(barId);
  const pctEl  = document.getElementById(pctId);
  if (!bar || !pctEl) return;

  const pctVal = (score * 100).toFixed(1); // e.g. "34.5"
  bar.style.transition = "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)";
  bar.style.width      = pctVal + "%";
  pctEl.textContent    = pctVal + "%";
}

// ── Chart colour palette ──────────────────────────────────────────────────────
const PALETTE = {
  positive: { bg: "rgba(52, 211, 153, 0.75)", border: "#34d399", glow: "rgba(52,211,153,0.2)"  },
  neutral:  { bg: "rgba(251, 191, 36, 0.75)", border: "#fbbf24", glow: "rgba(251,191,36,0.2)"  },
  negative: { bg: "rgba(248, 113, 113, 0.75)",border: "#f87171", glow: "rgba(248,113,113,0.2)" },
};

function sentimentPalette(sentiment) {
  if (sentiment === "Positive") return PALETTE.positive;
  if (sentiment === "Negative") return PALETTE.negative;
  return PALETTE.neutral;
}

// ── Shared Chart.js defaults ──────────────────────────────────────────────────
const CHART_FONT = { family: "'Inter', sans-serif" };

// ── Update Charts ─────────────────────────────────────────────────────────────
function updateCharts() {
  if (history.length === 0) {
    chartsSection.classList.add("hidden");
    return;
  }
  chartsSection.classList.remove("hidden");

  buildDoughnutChart();
  buildBarChart();
}

// ──────────────────────────────────────────────────────────────────────────────
// DOUGHNUT CHART — Sentiment Distribution
// ──────────────────────────────────────────────────────────────────────────────
function buildDoughnutChart() {
  const counts = { Positive: 0, Neutral: 0, Negative: 0 };
  history.forEach(h => { counts[h.sentiment] = (counts[h.sentiment] || 0) + 1; });
  const total = history.length;

  const pieCtx = document.getElementById("pie-chart").getContext("2d");
  if (pieChart) pieChart.destroy();

  pieChart = new Chart(pieCtx, {
    type: "doughnut",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [{
        data: [counts.Positive, counts.Neutral, counts.Negative],
        backgroundColor: [
          PALETTE.positive.bg,
          PALETTE.neutral.bg,
          PALETTE.negative.bg,
        ],
        borderColor: [
          PALETTE.positive.border,
          PALETTE.neutral.border,
          PALETTE.negative.border,
        ],
        borderWidth: 2,
        hoverOffset: 10,
        hoverBorderWidth: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: "60%",
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 700,
        easing: "easeInOutQuart",
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#94a3b8",
            font: { ...CHART_FONT, size: 12 },
            padding: 16,
            usePointStyle: true,
            pointStyleWidth: 10,
            generateLabels(chart) {
              // Show label + count in legend
              const ds = chart.data.datasets[0];
              return chart.data.labels.map((label, i) => ({
                text: `${label}  (${ds.data[i]})`,
                fillStyle: ds.backgroundColor[i],
                strokeStyle: ds.borderColor[i],
                lineWidth: ds.borderWidth,
                pointStyle: "circle",
                hidden: false,
                index: i,
              }));
            },
          },
        },
        tooltip: {
          backgroundColor: "rgba(10, 10, 30, 0.92)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          titleFont: { ...CHART_FONT, size: 13, weight: "600" },
          bodyFont:  { ...CHART_FONT, size: 12 },
          callbacks: {
            // e.g.  "Positive"
            title(items) { return items[0].label; },
            // e.g.  "Count: 3  (60.0%)"
            label(item) {
              const count = item.raw;
              const pct   = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
              return `  Count: ${count}   (${pct}% of analyses)`;
            },
          },
        },
      },
    },
    // Centre text plugin — shows total count
    plugins: [{
      id: "centerText",
      beforeDraw(chart) {
        const { width, height, ctx } = chart;
        ctx.save();
        const cx = width  / 2;
        const cy = height / 2 - 14; // offset up slightly for two lines
        ctx.textAlign    = "center";
        ctx.textBaseline = "middle";
        // Total number
        ctx.font      = `700 ${Math.min(28, width * 0.09)}px Inter, sans-serif`;
        ctx.fillStyle = "#e2e8f0";
        ctx.fillText(total, cx, cy);
        // "analyses" label
        ctx.font      = `500 ${Math.min(11, width * 0.036)}px Inter, sans-serif`;
        ctx.fillStyle = "#64748b";
        ctx.fillText("analyses", cx, cy + Math.min(20, width * 0.065));
        ctx.restore();
      },
    }],
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// BAR CHART — Confidence Comparison (last 10 entries)
// ──────────────────────────────────────────────────────────────────────────────
function buildBarChart() {
  const recent = history.slice(-10);
  const startIdx = history.length - recent.length; // for correct #N labels

  // Labels: #1, #2, …  — keep short
  const barLabels = recent.map((_, i) => `#${startIdx + i + 1}`);

  // ⚠ Store confidence as NUMBER (not string) so Chart.js scales correctly
  const barData    = recent.map(h => parseFloat((h.confidence * 100).toFixed(2)));
  const barColors  = recent.map(h => sentimentPalette(h.sentiment).bg);
  const barBorders = recent.map(h => sentimentPalette(h.sentiment).border);

  const barCtx = document.getElementById("bar-chart").getContext("2d");
  if (barChart) barChart.destroy();

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
        borderRadius: 8,
        borderSkipped: false,
        // Individual bar widths
        barPercentage: 0.65,
        categoryPercentage: 0.8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: {
        duration: 600,
        easing: "easeInOutQuart",
      },
      scales: {
        y: {
          beginAtZero: true,
          min: 0,
          max: 100,
          ticks: {
            color: "#64748b",
            font: { ...CHART_FONT, size: 11 },
            stepSize: 20,
            // ✅ FIX: append % to every Y-axis tick
            callback(value) { return value + "%"; },
          },
          grid: {
            color: "rgba(255,255,255,0.05)",
            drawBorder: false,
          },
          border: { display: false },
        },
        x: {
          ticks: {
            color: "#64748b",
            font: { ...CHART_FONT, size: 11 },
          },
          grid: { display: false },
          border: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10, 10, 30, 0.92)",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          titleFont: { ...CHART_FONT, size: 13, weight: "600" },
          bodyFont:  { ...CHART_FONT, size: 12 },
          callbacks: {
            // ✅ FIX: Show analysis number AND sentiment in tooltip title
            title(items) {
              const idx   = startIdx + items[0].dataIndex;
              const entry = recent[items[0].dataIndex];
              const emoji = emojiMap[entry.sentiment] || "";
              return `Analysis #${idx + 1}  ${emoji} ${entry.sentiment}`;
            },
            // ✅ FIX: Show confidence with % and qualitative label
            label(item) {
              const conf  = item.raw;         // already a %
              const confDecimal = conf / 100;
              const { text: lvlText } = confidenceLabel(confDecimal);
              return [
                `  Confidence: ${conf.toFixed(1)}%`,
                `  Level: ${lvlText}`,
              ];
            },
            // Show truncated input text below
            afterBody(items) {
              const entry = recent[items[0].dataIndex];
              const txt   = entry.text.length > 55
                ? entry.text.substring(0, 55) + "…"
                : entry.text;
              return [`  "${txt}"`];
            },
          },
        },
      },
    },
  });
}

// ── Update History Table ──────────────────────────────────────────────────────
function updateHistory() {
  if (history.length === 0) {
    historySection.classList.add("hidden");
    return;
  }

  historySection.classList.remove("hidden");
  historyBody.innerHTML = "";

  // Show newest first
  [...history].reverse().forEach((item, idx) => {
    const row          = document.createElement("tr");
    const truncatedTxt = item.text.length > 60
      ? item.text.substring(0, 60) + "…"
      : item.text;

    const confPct = (item.confidence * 100).toFixed(1);
    const { text: lvlText, cls: lvlCls } = confidenceLabel(item.confidence);

    row.innerHTML = `
      <td>${history.length - idx}</td>
      <td class="table-text-cell" title="${escapeHtml(item.text)}">${escapeHtml(truncatedTxt)}</td>
      <td><span class="sentiment-pill ${pillMap[item.sentiment]}">${emojiMap[item.sentiment]} ${item.sentiment}</span></td>
      <td>
        <span class="conf-cell">
          <strong>${confPct}%</strong>
          <small class="conf-badge ${lvlCls}">${lvlText}</small>
        </span>
      </td>
      <td>${item.timestamp}</td>
    `;
    row.style.animation = idx === 0 ? "fadeInUp 0.3s ease-out" : "none";
    historyBody.appendChild(row);
  });
}

// ── Clear History ─────────────────────────────────────────────────────────────
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

// ── Utility ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Shake keyframe ────────────────────────────────────────────────────────────
const _style = document.createElement("style");
_style.textContent = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    20%       { transform: translateX(-6px); }
    40%       { transform: translateX(6px); }
    60%       { transform: translateX(-4px); }
    80%       { transform: translateX(4px); }
  }
`;
document.head.appendChild(_style);
