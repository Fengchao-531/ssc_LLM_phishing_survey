const detectorSelect = document.querySelector("#detectorSelect");
const stageSelect = document.querySelector("#stageSelect");
const modeSelect = document.querySelector("#modeSelect");
const metricCards = document.querySelector("#metricCards");
const metricTable = document.querySelector("#metricTable");
const figures = document.querySelector("#figures");

const formatName = (name) => name.replaceAll("_", " ");
const pct = (value) => value == null ? "NA" : `${(value * 100).toFixed(1)}%`;
const dec = (value) => value == null ? "NA" : value.toFixed(3);
const rows = ["LLM", "HW"];

function imageSet(detector, stage) {
  const base = `output/${detector}/${stage}`;
  if (stage === "overview") {
    return [
      ["Surrogate response map", `${base}/surrogate_response_tp_fn_map.png`],
      ["Overview persuasion heatmap", `${base}/group1_difference_heatmaps.png`],
    ];
  }
  if (stage === "S8") {
    return [
      ["Surrogate response map, row 1", `${base}/surrogate_response_tp_fn_map_row1.png`],
      ["Surrogate response map, row 2", `${base}/surrogate_response_tp_fn_map_row2.png`],
      ["FN persuasion boxplots, row 1", `${base}/fn_selected_group_boxplots_row1.png`],
      ["FN persuasion boxplots, row 2", `${base}/fn_selected_group_boxplots_row2.png`],
    ];
  }
  return [
    ["Surrogate response map", `${base}/surrogate_response_tp_fn_map.png`],
    ["FN persuasion boxplots", `${base}/fn_selected_group_boxplots.png`],
  ];
}

function renderCards(stageMetrics, mode) {
  const llm = stageMetrics.LLM || {};
  const cards = mode === "recall"
    ? [
        ["LLM Recall", pct(llm.recall)],
        ["LLM FNR", pct(llm.fnr)],
        ["Rows", llm.n_rows?.toLocaleString() ?? "NA"],
      ]
    : [
        ["MCC", dec(llm.mcc)],
        ["Recall", pct(llm.recall)],
        ["TNR", pct(llm.tnr)],
      ];
  metricCards.innerHTML = cards.map(([label, value]) => `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
    </article>
  `).join("");
}

function renderTable(stageMetrics, mode) {
  const headers = mode === "recall"
    ? ["Dataset", "Recall", "FNR", "Precision", "Rows"]
    : ["Dataset", "MCC", "Recall", "TNR", "Precision", "F1", "Accuracy", "Rows"];
  const body = rows.map((label) => {
    const item = stageMetrics[label] || {};
    const cells = mode === "recall"
      ? [label, pct(item.recall), pct(item.fnr), pct(item.precision), item.n_rows?.toLocaleString() ?? "NA"]
      : [label, dec(item.mcc), pct(item.recall), pct(item.tnr), pct(item.precision), pct(item.f1), pct(item.accuracy), item.n_rows?.toLocaleString() ?? "NA"];
    return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
  }).join("");
  metricTable.innerHTML = `
    <thead><tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr></thead>
    <tbody>${body}</tbody>
  `;
}

function renderFigures(detector, stage) {
  figures.innerHTML = imageSet(detector, stage).map(([title, src]) => `
    <article class="figure-panel">
      <h2>${title}</h2>
      <img src="${src}" alt="${title} for ${detector} ${stage}">
    </article>
  `).join("");
}

function render(data) {
  const detector = detectorSelect.value || "scamllm";
  const stage = stageSelect.value || "overview";
  const mode = modeSelect.value || "metrics";
  const stageMetrics = data.metrics[detector]?.[stage] || {};
  renderCards(stageMetrics, mode);
  renderTable(stageMetrics, mode);
  renderFigures(detector, stage);
}

fetch("data/detector_metrics.json")
  .then((response) => response.json())
  .then((data) => {
    detectorSelect.innerHTML = data.detectors.map((detector) => (
      `<option value="${detector}">${formatName(detector)}</option>`
    )).join("");
    detectorSelect.value = "scamllm";
    stageSelect.value = "overview";
    modeSelect.value = "metrics";
    [detectorSelect, stageSelect, modeSelect].forEach((control) => {
      control.addEventListener("change", () => render(data));
    });
    render(data);
  });
