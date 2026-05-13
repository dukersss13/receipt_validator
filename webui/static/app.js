const state = {
    validatedTransactions: [],
    discrepancies: [],
    unmatchedTransactions: [],
    unmatchedProofs: [],
    recommendations: [],
    loadedTransactions: [],
    loadedProofs: [],
    chatHistory: [],
    chatIsStreaming: false,
};

let progressTimer = null;
let progressValue = 0;
let isValidationRunning = false;
let greetingTimer = null;
const dataSourcePanelState = {
    transactionsCollapsed: true,
    proofsCollapsed: true,
};

const BUFFERING_PHRASES = [
    "Let me see...",
    "Let me look into it...",
    "Checking your transactions...",
    "One moment...",
    "Looking that up...",
    "Crunching the numbers...",
];

function randomBufferingPhrase() {
    return BUFFERING_PHRASES[Math.floor(Math.random() * BUFFERING_PHRASES.length)];
}

const byId = (id) => document.getElementById(id);

function buildLocalSessionId() {
    if (globalThis.crypto?.randomUUID) {
        return globalThis.crypto.randomUUID();
    }

    return `local-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function setStatus(text) {
    byId("workflow-status").value = text;
}

function updateValidationStageStatus() {
    if (!isValidationRunning) {
        return;
    }

    if (progressValue < 35) {
        setStatus("Reading transactions...");
    } else if (progressValue < 70) {
        setStatus("Reading proofs...");
    } else if (progressValue < 90) {
        setStatus("Matching records...");
    } else {
        setStatus("Finalizing results...");
    }
}

function showError(message) {
    const error = byId("error-text");
    if (!message) {
        error.hidden = true;
        error.textContent = "";
        return;
    }

    error.hidden = false;
    error.textContent = message;
}

function setProgress(value) {
    const clamped = Math.max(0, Math.min(100, Math.round(value)));
    progressValue = clamped;

    byId("progress-fill").style.width = `${clamped}%`;
    byId("progress-track").setAttribute("aria-valuenow", String(clamped));
    byId("progress-label").textContent = `${clamped}%`;
}

function setLoadingState(isLoading) {
    const progressTrack = byId("progress-track");
    const progressFill = byId("progress-fill");
    const receiptLoader = byId("receipt-loader");

    progressTrack.classList.toggle("loading", isLoading);
    progressFill.classList.toggle("loading", isLoading);
    receiptLoader.classList.toggle("loading", isLoading);
    receiptLoader.hidden = !isLoading;
}

function setProgressShellVisible(isVisible) {
    byId("progress-shell").hidden = !isVisible;
}

function startProgress() {
    clearInterval(progressTimer);
    setProgressShellVisible(true);
    setLoadingState(true);
    setProgress(4);
    updateValidationStageStatus();

    progressTimer = setInterval(() => {
        if (progressValue >= 99) {
            return;
        }

        const remaining = 99 - progressValue;
        const maxStep = progressValue < 70 ? 4 : (progressValue < 90 ? 2 : 1);
        const increment = Math.max(
            1,
            Math.min(remaining, 1 + Math.floor(Math.random() * maxStep)),
        );
        setProgress(progressValue + increment);
        updateValidationStageStatus();
    }, 420);
}

function completeProgress() {
    clearInterval(progressTimer);
    setProgress(100);
    setLoadingState(false);
}

function resetProgress() {
    clearInterval(progressTimer);
    setProgress(0);
    setLoadingState(false);
    setProgressShellVisible(false);
}

function renderTable(elementId, rows) {
    const container = byId(elementId);
    container.innerHTML = "";

    if (!rows || rows.length === 0) {
        const empty = document.createElement("div");
        empty.className = "table-empty";
        empty.textContent = "No rows";
        container.appendChild(empty);
        return;
    }

    const columns = [];
    rows.forEach((row) => {
        Object.keys(row).forEach((key) => {
            if (!columns.includes(key)) {
                columns.push(key);
            }
        });
    });
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        headRow.appendChild(th);
    });

    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
        const tr = document.createElement("tr");
        columns.forEach((col) => {
            const td = document.createElement("td");
            td.textContent = row[col] ?? "";
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
}

function setDataSourceCollapsed(kind, collapsed) {
    if (kind === "transactions") {
        dataSourcePanelState.transactionsCollapsed = collapsed;
    } else {
        dataSourcePanelState.proofsCollapsed = collapsed;
    }
    updateDataSourceCard(kind);
}

function updateDataSourceCard(kind) {
    const isTransactions = kind === "transactions";
    const rows = isTransactions ? state.loadedTransactions : state.loadedProofs;
    const bodyId = isTransactions ? "loaded-transactions-body" : "loaded-proofs-body";
    const countId = isTransactions ? "loaded-transactions-count" : "loaded-proofs-count";
    const toggleId = isTransactions ? "loaded-transactions-toggle" : "loaded-proofs-toggle";
    const collapsed = isTransactions
        ? dataSourcePanelState.transactionsCollapsed
        : dataSourcePanelState.proofsCollapsed;

    const body = byId(bodyId);
    const count = byId(countId);
    const toggle = byId(toggleId);

    if (count) {
        count.textContent = `${rows.length} rows`;
    }
    if (body) {
        body.classList.toggle("collapsed", collapsed);
    }
    if (toggle) {
        toggle.textContent = collapsed ? "Expand" : "Collapse";
        toggle.setAttribute("aria-expanded", String(!collapsed));
    }
}

function renderDataSourceTables() {
    renderTable("loaded-transactions-table", state.loadedTransactions);
    renderTable("loaded-proofs-table", state.loadedProofs);
    updateDataSourceCard("transactions");
    updateDataSourceCard("proofs");
}

function promptRerunAfterAdd() {
    const txCount = byId("transactions-input")?.files?.length ?? 0;
    const proofCount = byId("proofs-input")?.files?.length ?? 0;

    if (txCount === 0 && proofCount === 0) {
        return;
    }

    if (txCount === 0 || proofCount === 0) {
        showError(
            "Add files to both Transactions and Proofs, then re-run validation to refresh results.",
        );
        return;
    }

    const shouldRerun = globalThis.confirm(
        "New files were added. Re-run validation now to update results?",
    );
    if (shouldRerun) {
        runValidation();
    }
}

function setPanelVisibility(panelId, visible) {
    const panel = byId(panelId);
    if (!panel) {
        return;
    }

    panel.classList.toggle("hidden-panel", !visible);
}

function hasMeaningfulRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        return false;
    }

    return rows.some((row) => {
        if (!row || typeof row !== "object") {
            return false;
        }

        return Object.values(row).some(
            (value) => value !== null && value !== undefined && String(value).trim() !== "",
        );
    });
}

function updateResultPanelsVisibility() {
    setPanelVisibility("loaded-transactions-panel", true);
    setPanelVisibility("loaded-proofs-panel", true);
    setPanelVisibility("validated-panel", state.validatedTransactions.length > 0);
    setPanelVisibility("discrepancies-panel", state.discrepancies.length > 0);
    setPanelVisibility(
        "unmatched-transactions-panel",
        state.unmatchedTransactions.length > 0,
    );
    setPanelVisibility("unmatched-proofs-panel", state.unmatchedProofs.length > 0);
    setPanelVisibility("recommendations-panel", state.recommendations.length > 0);
}

function renderChatTranscript() {
    const host = byId("chat-transcript");
    if (!host) {
        return;
    }

    host.innerHTML = "";
    const rows = state.chatHistory || [];

    if (!rows.length) {
        const welcome = document.createElement("div");
        welcome.className = "chat-welcome";
        welcome.innerHTML =
            "<strong>Hi, I\u2019m ArVee!</strong>" +
            "<span>I\u2019m here to help with your finances. Try asking me questions like:</span>" +
            '<ul class="chat-welcome-examples">' +
            '<li data-q="How much did I spend on food this month?">\u201cHow much did I spend on food this month?\u201d</li>' +
            '<li data-q="What category did I spend the most money on?">\u201cWhat category did I spend the most money on?\u201d</li>' +
            '<li data-q="Show my top 5 spending categories">\u201cShow my top 5 spending categories\u201d</li>' +
            '<li data-q="Show me a chart of my spendings this month">\u201cShow me a chart of my spendings this month\u201d</li>' +
            "</ul>";
        welcome.querySelectorAll(".chat-welcome-examples li").forEach((li) => {
            li.addEventListener("click", () => {
                const input = byId("chat-input");
                if (input) {
                    input.value = li.dataset.q;
                    sendChatMessage();
                }
            });
        });
        host.appendChild(welcome);
        return;
    }

    rows.forEach((msg) => {
        const item = document.createElement("div");
        item.className = `chat-msg chat-msg-${msg.role}`;
        if (msg.pending) {
            item.classList.add("chat-msg-pending");
        }

        if (msg.role === "assistant" && msg.chart) {
            const chartNode = buildChatChartNode(msg.chart);
            if (chartNode) {
                item.appendChild(chartNode);
            }
        }

        const text = document.createElement("div");
        text.className = "chat-msg-text";
        text.textContent = msg.text;
        item.appendChild(text);

        if (msg.role === "assistant" && msg.top_categories) {
            const tableNode = buildTopCategoriesTable(msg.top_categories);
            if (tableNode) {
                item.appendChild(tableNode);
            }
        }

        if (msg.role === "assistant" && msg.comparison_table) {
            const tableNode = buildComparisonTable(msg.comparison_table);
            if (tableNode) {
                item.appendChild(tableNode);
            }
        }

        host.appendChild(item);
    });

    host.scrollTop = host.scrollHeight;
}

function formatUsd(value) {
    const numeric = Number(value) || 0;
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(numeric);
}

function buildTopCategoriesTable(categories) {
    if (!Array.isArray(categories) || !categories.length) {
        return null;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "chat-data-table";

    const heading = document.createElement("div");
    heading.className = "chat-data-table-title";
    heading.textContent = "Top 5 Categories";
    wrapper.appendChild(heading);

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Category</th><th>Amount</th></tr>";
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    categories.forEach((row) => {
        const tr = document.createElement("tr");
        const tdCat = document.createElement("td");
        tdCat.textContent = row.category || "";
        const tdVal = document.createElement("td");
        tdVal.className = "num";
        tdVal.textContent = formatUsd(row.value);
        tr.appendChild(tdCat);
        tr.appendChild(tdVal);
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
}

function buildComparisonTable(tableData) {
    if (!tableData || !Array.isArray(tableData.rows) || !tableData.rows.length) {
        return null;
    }

    const columns = Array.isArray(tableData.columns) ? tableData.columns : [];
    const wrapper = document.createElement("div");
    wrapper.className = "chat-data-table";

    const heading = document.createElement("div");
    heading.className = "chat-data-table-title";
    heading.textContent = "Spending Comparison";
    wrapper.appendChild(heading);

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    tableData.rows.forEach((row) => {
        const tr = document.createElement("tr");

        const tdCat = document.createElement("td");
        tdCat.textContent = row.category || "";
        tr.appendChild(tdCat);

        const tdP1 = document.createElement("td");
        tdP1.className = "num";
        tdP1.textContent = formatUsd(row.period_1);
        tr.appendChild(tdP1);

        const tdP2 = document.createElement("td");
        tdP2.className = "num";
        tdP2.textContent = formatUsd(row.period_2);
        tr.appendChild(tdP2);

        const tdDelta = document.createElement("td");
        tdDelta.className = "num";
        const delta = Number(row.delta) || 0;
        tdDelta.textContent = (delta >= 0 ? "+" : "") + formatUsd(delta);
        tdDelta.classList.add(delta >= 0 ? "positive" : "negative");
        tr.appendChild(tdDelta);

        const tdPct = document.createElement("td");
        tdPct.className = "num";
        if (row.percent_change != null) {
            const pct = Number(row.percent_change);
            tdPct.textContent = (pct >= 0 ? "+" : "") + pct.toFixed(1) + "%";
            tdPct.classList.add(pct >= 0 ? "positive" : "negative");
        } else {
            tdPct.textContent = "—";
        }
        tr.appendChild(tdPct);

        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
}

function buildChatChartNode(chart) {
    if (!chart) {
        return null;
    }

    let inner = null;
    if (chart.type === "grouped_bar") {
        inner = buildGroupedBarChartNode(chart);
    } else if (chart.type === "bar") {
        inner = buildSingleBarChartNode(chart);
    } else if (chart.type === "pie") {
        inner = buildPieChartNode(chart);
    }

    if (!inner) {
        return null;
    }

    const expandBtn = document.createElement("button");
    expandBtn.className = "chat-chart-expand-btn";
    expandBtn.setAttribute("aria-label", "Expand chart");
    expandBtn.setAttribute("title", "Expand chart");
    expandBtn.innerHTML =
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">' +
        '<path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
        "</svg>";
    expandBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openChartModal(chart);
    });
    inner.appendChild(expandBtn);

    return inner;
}

function openChartModal(chart) {
    let overlay = byId("chart-modal-overlay");
    if (overlay) {
        overlay.remove();
    }

    overlay = document.createElement("div");
    overlay.id = "chart-modal-overlay";
    overlay.className = "chart-modal-overlay";

    const modal = document.createElement("div");
    modal.className = "chart-modal";

    const closeBtn = document.createElement("button");
    closeBtn.className = "chart-modal-close-btn";
    closeBtn.setAttribute("aria-label", "Close");
    closeBtn.innerHTML = "&times;";
    closeBtn.addEventListener("click", () => overlay.remove());
    modal.appendChild(closeBtn);

    let expanded = null;
    if (chart.type === "grouped_bar") {
        expanded = buildGroupedBarChartNode(chart);
    } else if (chart.type === "bar") {
        expanded = buildSingleBarChartNode(chart);
    } else if (chart.type === "pie") {
        expanded = buildPieChartNode(chart);
    }

    if (expanded) {
        expanded.classList.add("chat-chart-expanded");
        modal.appendChild(expanded);
    }

    overlay.appendChild(modal);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) {
            overlay.remove();
        }
    });
    document.addEventListener("keydown", function escHandler(e) {
        if (e.key === "Escape" && byId("chart-modal-overlay")) {
            overlay.remove();
            document.removeEventListener("keydown", escHandler);
        }
    });

    document.body.appendChild(overlay);
}

function buildGroupedBarChartNode(chart) {

    const categories = Array.isArray(chart.x) ? chart.x.map((v) => String(v || "")) : [];
    const series = Array.isArray(chart.series) ? chart.series : [];
    if (!categories.length || !series.length) {
        return null;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "chat-chart";

    const title = document.createElement("div");
    title.className = "chat-chart-title";
    title.textContent = chart.title || "Category comparison";
    wrapper.appendChild(title);

    const tooltip = document.createElement("div");
    tooltip.className = "chat-chart-tooltip";
    tooltip.hidden = true;
    wrapper.appendChild(tooltip);

    const legend = document.createElement("div");
    legend.className = "chat-chart-legend";
    series.forEach((s, idx) => {
        const legendItem = document.createElement("span");
        legendItem.className = "chat-chart-legend-item";

        const swatch = document.createElement("span");
        swatch.className = "chat-chart-swatch";
        swatch.style.backgroundColor = idx === 0 ? "#0f7b6c" : "#e59f3a";

        const label = document.createElement("span");
        label.textContent = String(s.name || `Series ${idx + 1}`);

        legendItem.appendChild(swatch);
        legendItem.appendChild(label);
        legend.appendChild(legendItem);
    });
    wrapper.appendChild(legend);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 420 250");
    svg.setAttribute("class", "chat-chart-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", chart.title || "Grouped bar chart");

    const width = 420;
    const height = 250;
    const margin = { top: 14, right: 12, bottom: 70, left: 46 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const values = series.flatMap((s) =>
        Array.isArray(s.values) ? s.values.map((value) => Number(value) || 0) : [],
    );
    const maxValue = Math.max(1, ...values);

    const axis = document.createElementNS("http://www.w3.org/2000/svg", "path");
    axis.setAttribute(
        "d",
        `M ${margin.left} ${margin.top} V ${margin.top + plotHeight} H ${margin.left + plotWidth}`,
    );
    axis.setAttribute("stroke", "#b8ad99");
    axis.setAttribute("stroke-width", "1");
    axis.setAttribute("fill", "none");
    svg.appendChild(axis);

    const tickCount = 4;
    for (let i = 0; i <= tickCount; i += 1) {
        const value = (maxValue / tickCount) * i;
        const y = margin.top + plotHeight - (value / maxValue) * plotHeight;

        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", String(margin.left));
        grid.setAttribute("x2", String(margin.left + plotWidth));
        grid.setAttribute("y1", String(y));
        grid.setAttribute("y2", String(y));
        grid.setAttribute("stroke", i === 0 ? "#c8beac" : "#ece4d6");
        grid.setAttribute("stroke-width", "1");
        svg.appendChild(grid);

        const tick = document.createElementNS("http://www.w3.org/2000/svg", "text");
        tick.setAttribute("x", String(margin.left - 8));
        tick.setAttribute("y", String(y + 4));
        tick.setAttribute("text-anchor", "end");
        tick.setAttribute("class", "chat-chart-axis-text");
        tick.textContent = formatUsd(value);
        svg.appendChild(tick);
    }

    const categoryBand = plotWidth / categories.length;
    const groupWidth = Math.min(46, categoryBand * 0.72);
    const barGap = 3;
    const barWidth = Math.max(
        5,
        (groupWidth - barGap * (series.length - 1)) / Math.max(series.length, 1),
    );

    categories.forEach((category, categoryIndex) => {
        const groupStartX =
            margin.left + categoryIndex * categoryBand + (categoryBand - groupWidth) / 2;

        series.forEach((entry, seriesIndex) => {
            const value = Number(entry?.values?.[categoryIndex] ?? 0) || 0;
            const barHeight = (value / maxValue) * plotHeight;
            const x = groupStartX + seriesIndex * (barWidth + barGap);
            const y = margin.top + plotHeight - barHeight;

            const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            bar.setAttribute("x", String(x));
            bar.setAttribute("y", String(y));
            bar.setAttribute("width", String(barWidth));
            bar.setAttribute("height", String(Math.max(0, barHeight)));
            bar.setAttribute("rx", "2");
            bar.setAttribute("fill", seriesIndex === 0 ? "#0f7b6c" : "#e59f3a");
            bar.classList.add("chat-chart-bar");

            const label = String(entry?.name || `Series ${seriesIndex + 1}`);
            const tooltipText = `${label}: ${formatUsd(value)}`;

            bar.addEventListener("mousemove", (event) => {
                const bounds = wrapper.getBoundingClientRect();
                tooltip.hidden = false;
                tooltip.textContent = tooltipText;
                tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
                tooltip.style.top = `${event.clientY - bounds.top - 18}px`;
            });
            bar.addEventListener("mouseleave", () => {
                tooltip.hidden = true;
            });

            const nativeTooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
            nativeTooltip.textContent = tooltipText;
            bar.appendChild(nativeTooltip);
            svg.appendChild(bar);
        });

        const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const labelX = groupStartX + groupWidth / 2;
        const labelY = margin.top + plotHeight + 16;
        xLabel.setAttribute("x", String(labelX));
        xLabel.setAttribute("y", String(labelY));
        xLabel.setAttribute("class", "chat-chart-axis-text");
        const maxLen = categories.length > 4 ? 9 : 12;
        xLabel.textContent = category.length > maxLen ? `${category.slice(0, maxLen)}…` : category;
        if (categories.length > 4) {
            xLabel.setAttribute("text-anchor", "end");
            xLabel.setAttribute("transform", `rotate(-40, ${labelX}, ${labelY})`);
        } else {
            xLabel.setAttribute("text-anchor", "middle");
        }
        svg.appendChild(xLabel);
    });

    wrapper.appendChild(svg);
    return wrapper;
}

function buildSingleBarChartNode(chart) {
    const labels = Array.isArray(chart.labels) ? chart.labels.map((v) => String(v || "")) : [];
    const values = Array.isArray(chart.values) ? chart.values.map((v) => Number(v) || 0) : [];
    if (!labels.length || labels.length !== values.length) {
        return null;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "chat-chart";

    const title = document.createElement("div");
    title.className = "chat-chart-title";
    title.textContent = chart.title || "Spending breakdown";
    wrapper.appendChild(title);

    const tooltip = document.createElement("div");
    tooltip.className = "chat-chart-tooltip";
    tooltip.hidden = true;
    wrapper.appendChild(tooltip);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 420 250");
    svg.setAttribute("class", "chat-chart-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", chart.title || "Bar chart");

    const width = 420;
    const height = 250;
    const margin = { top: 14, right: 12, bottom: 70, left: 46 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maxValue = Math.max(1, ...values);

    const axis = document.createElementNS("http://www.w3.org/2000/svg", "path");
    axis.setAttribute(
        "d",
        `M ${margin.left} ${margin.top} V ${margin.top + plotHeight} H ${margin.left + plotWidth}`,
    );
    axis.setAttribute("stroke", "#b8ad99");
    axis.setAttribute("stroke-width", "1");
    axis.setAttribute("fill", "none");
    svg.appendChild(axis);

    const tickCount = 4;
    for (let i = 0; i <= tickCount; i += 1) {
        const value = (maxValue / tickCount) * i;
        const y = margin.top + plotHeight - (value / maxValue) * plotHeight;

        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", String(margin.left));
        grid.setAttribute("x2", String(margin.left + plotWidth));
        grid.setAttribute("y1", String(y));
        grid.setAttribute("y2", String(y));
        grid.setAttribute("stroke", i === 0 ? "#c8beac" : "#ece4d6");
        grid.setAttribute("stroke-width", "1");
        svg.appendChild(grid);

        const tick = document.createElementNS("http://www.w3.org/2000/svg", "text");
        tick.setAttribute("x", String(margin.left - 8));
        tick.setAttribute("y", String(y + 4));
        tick.setAttribute("text-anchor", "end");
        tick.setAttribute("class", "chat-chart-axis-text");
        tick.textContent = formatUsd(value);
        svg.appendChild(tick);
    }

    const band = plotWidth / labels.length;
    const barWidth = Math.min(28, band * 0.66);
    labels.forEach((label, idx) => {
        const value = values[idx];
        const barHeight = (value / maxValue) * plotHeight;
        const x = margin.left + idx * band + (band - barWidth) / 2;
        const y = margin.top + plotHeight - barHeight;

        const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        bar.setAttribute("x", String(x));
        bar.setAttribute("y", String(y));
        bar.setAttribute("width", String(barWidth));
        bar.setAttribute("height", String(Math.max(0, barHeight)));
        bar.setAttribute("rx", "2");
        bar.setAttribute("fill", "#0f7b6c");
        bar.classList.add("chat-chart-bar");

        const tooltipText = `${label}: ${formatUsd(value)}`;
        bar.addEventListener("mousemove", (event) => {
            const bounds = wrapper.getBoundingClientRect();
            tooltip.hidden = false;
            tooltip.textContent = tooltipText;
            tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
            tooltip.style.top = `${event.clientY - bounds.top - 18}px`;
        });
        bar.addEventListener("mouseleave", () => {
            tooltip.hidden = true;
        });

        const nativeTooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
        nativeTooltip.textContent = tooltipText;
        bar.appendChild(nativeTooltip);
        svg.appendChild(bar);

        const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const labelX = x + barWidth / 2;
        const labelY = margin.top + plotHeight + 16;
        xLabel.setAttribute("x", String(labelX));
        xLabel.setAttribute("y", String(labelY));
        xLabel.setAttribute("class", "chat-chart-axis-text");
        const maxLen = labels.length > 4 ? 9 : 12;
        xLabel.textContent = label.length > maxLen ? `${label.slice(0, maxLen)}…` : label;
        if (labels.length > 4) {
            xLabel.setAttribute("text-anchor", "end");
            xLabel.setAttribute("transform", `rotate(-40, ${labelX}, ${labelY})`);
        } else {
            xLabel.setAttribute("text-anchor", "middle");
        }
        svg.appendChild(xLabel);
    });

    wrapper.appendChild(svg);
    return wrapper;
}

function buildPieChartNode(chart) {
    const labels = Array.isArray(chart.labels) ? chart.labels.map((v) => String(v || "")) : [];
    const values = Array.isArray(chart.values) ? chart.values.map((v) => Math.max(0, Number(v) || 0)) : [];
    if (!labels.length || labels.length !== values.length) {
        return null;
    }

    const total = values.reduce((acc, val) => acc + val, 0);
    if (total <= 0) {
        return null;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "chat-chart";

    const title = document.createElement("div");
    title.className = "chat-chart-title";
    title.textContent = chart.title || "Spending share by category";
    wrapper.appendChild(title);

    const legend = document.createElement("div");
    legend.className = "chat-chart-legend";
    wrapper.appendChild(legend);

    const tooltip = document.createElement("div");
    tooltip.className = "chat-chart-tooltip";
    tooltip.hidden = true;
    wrapper.appendChild(tooltip);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 420 250");
    svg.setAttribute("class", "chat-chart-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", chart.title || "Pie chart");

    const cx = 130;
    const cy = 130;
    const radius = 82;
    const colors = ["#0f7b6c", "#e59f3a", "#6f8a3b", "#cf6b4d", "#4a7895", "#b7779f", "#8f6b4f"];

    let startAngle = -Math.PI / 2;
    labels.forEach((label, idx) => {
        const value = values[idx];
        const sweep = (value / total) * (Math.PI * 2);
        const endAngle = startAngle + sweep;
        const x1 = cx + radius * Math.cos(startAngle);
        const y1 = cy + radius * Math.sin(startAngle);
        const x2 = cx + radius * Math.cos(endAngle);
        const y2 = cy + radius * Math.sin(endAngle);
        const largeArc = sweep > Math.PI ? 1 : 0;
        const color = colors[idx % colors.length];

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute(
            "d",
            `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z`,
        );
        path.setAttribute("fill", color);
        path.classList.add("chat-chart-bar");

        const percent = ((value / total) * 100).toFixed(1);
        const tooltipText = `${label}: ${formatUsd(value)} (${percent}%)`;

        path.addEventListener("mousemove", (event) => {
            const bounds = wrapper.getBoundingClientRect();
            tooltip.hidden = false;
            tooltip.textContent = tooltipText;
            tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
            tooltip.style.top = `${event.clientY - bounds.top - 18}px`;
        });
        path.addEventListener("mouseleave", () => {
            tooltip.hidden = true;
        });

        const nativeTooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
        nativeTooltip.textContent = tooltipText;
        path.appendChild(nativeTooltip);
        svg.appendChild(path);

        const legendItem = document.createElement("span");
        legendItem.className = "chat-chart-legend-item";
        const swatch = document.createElement("span");
        swatch.className = "chat-chart-swatch";
        swatch.style.backgroundColor = color;
        const legendText = document.createElement("span");
        legendText.textContent = `${label} ${percent}%`;
        legendItem.appendChild(swatch);
        legendItem.appendChild(legendText);
        legend.appendChild(legendItem);

        startAngle = endAngle;
    });

    const hole = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    hole.setAttribute("cx", String(cx));
    hole.setAttribute("cy", String(cy));
    hole.setAttribute("r", "34");
    hole.setAttribute("fill", "#f8f4ec");
    svg.appendChild(hole);

    const centerText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    centerText.setAttribute("x", String(cx));
    centerText.setAttribute("y", String(cy + 4));
    centerText.setAttribute("text-anchor", "middle");
    centerText.setAttribute("class", "chat-chart-axis-text");
    centerText.textContent = "100%";
    svg.appendChild(centerText);

    wrapper.appendChild(svg);
    return wrapper;
}

function setChatPopupOpen(isOpen) {
    const popup = byId("chat-popup");
    const fab = byId("chat-fab");
    const greeting = byId("chat-greeting");
    if (!popup || !fab) {
        return;
    }

    popup.hidden = !isOpen;
    fab.setAttribute("aria-expanded", String(isOpen));

    if (greeting) {
        greeting.hidden = isOpen;
    }

    if (isOpen) {
        renderChatTranscript();
        byId("chat-input")?.focus();
    }
}

function startGreetingFlash() {
    const greeting = byId("chat-greeting");
    if (!greeting) {
        return;
    }

    const messages = [
        "Hi, I\u2019m ArVee! Ask me about your finances.",
        "Try: \u201cHow much did I spend on food?\u201d",
        "Try: \u201cWhat\u2019s my top spending category?\u201d",
        "I can answer questions about your transactions.",
    ];
    let index = 0;
    greeting.textContent = messages[index];

    clearInterval(greetingTimer);
    greetingTimer = setInterval(() => {
        greeting.style.opacity = "0";
        setTimeout(() => {
            index = (index + 1) % messages.length;
            greeting.textContent = messages[index];
            greeting.style.opacity = "";
        }, 500);
    }, 4500);
}

function renderDiscrepanciesTable() {
    const container = byId("discrepancies-table");
    container.innerHTML = "";

    const rows = state.discrepancies;
    if (!rows || rows.length === 0) {
        byId("validate-discrepancies-btn").disabled = true;
        const empty = document.createElement("div");
        empty.className = "table-empty";
        empty.textContent = "No rows";
        container.appendChild(empty);
        return;
    }

    const columns = Object.keys(rows[0]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    const selectHeader = document.createElement("th");
    selectHeader.className = "check-cell";
    selectHeader.textContent = "Select";
    headRow.appendChild(selectHeader);

    columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        headRow.appendChild(th);
    });

    const adjustHeader = document.createElement("th");
    adjustHeader.textContent = "Adjusted Amount";
    headRow.appendChild(adjustHeader);

    const commentHeader = document.createElement("th");
    commentHeader.textContent = "Comment";
    headRow.appendChild(commentHeader);

    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");

        const checkCell = document.createElement("td");
        checkCell.className = "check-cell";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "discrepancy-select";
        checkbox.value = String(idx);
        checkCell.appendChild(checkbox);
        tr.appendChild(checkCell);

        columns.forEach((col) => {
            const td = document.createElement("td");
            td.textContent = row[col] ?? "";
            tr.appendChild(td);
        });

        const adjustCell = document.createElement("td");
        const adjustInput = document.createElement("input");
        adjustInput.type = "number";
        adjustInput.step = "0.01";
        adjustInput.className = "table-input";
        adjustInput.id = `adjusted-amount-${idx}`;
        adjustInput.value = row["Transaction Total"] ?? row["Proof Total"] ?? "";
        adjustCell.appendChild(adjustInput);
        tr.appendChild(adjustCell);

        const commentCell = document.createElement("td");
        const commentInput = document.createElement("input");
        commentInput.type = "text";
        commentInput.className = "table-input";
        commentInput.id = `discrepancy-comment-${idx}`;
        commentInput.placeholder = "Optional comment";
        commentCell.appendChild(commentInput);
        tr.appendChild(commentCell);

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
    byId("validate-discrepancies-btn").disabled = false;
}

function renderRecommendationsTable() {
    const container = byId("recommendations-table");
    container.innerHTML = "";

    const rows = state.recommendations;
    if (!rows || rows.length === 0) {
        byId("accept-recommendations-btn").disabled = true;
        const empty = document.createElement("div");
        empty.className = "table-empty";
        empty.textContent = "No rows";
        container.appendChild(empty);
        return;
    }

    const columns = Object.keys(rows[0]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    const selectHeader = document.createElement("th");
    selectHeader.className = "check-cell";
    selectHeader.textContent = "Select";
    headRow.appendChild(selectHeader);

    columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        headRow.appendChild(th);
    });

    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");

        const checkCell = document.createElement("td");
        checkCell.className = "check-cell";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "recommendation-select";
        checkbox.value = String(idx);
        checkCell.appendChild(checkbox);
        tr.appendChild(checkCell);

        columns.forEach((col) => {
            const td = document.createElement("td");
            td.textContent = row[col] ?? "";
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
    byId("accept-recommendations-btn").disabled = false;
}

function getSelectedIndices(inputName) {
    return Array.from(document.querySelectorAll(`input[name='${inputName}']:checked`))
        .map((node) => Number(node.value))
        .filter((value) => Number.isInteger(value));
}

function updateManualMatchButtonState() {
    const button = byId("manual-match-btn");
    if (!button) {
        return;
    }

    const selectedTx = getSelectedIndices("unmatched-transaction-select");
    const selectedProofs = getSelectedIndices("unmatched-proof-select");

    button.disabled =
        selectedTx.length === 0 ||
        selectedProofs.length === 0 ||
        selectedTx.length !== selectedProofs.length;
}

function renderSelectableUnmatchedTable(elementId, rows, checkboxName) {
    const container = byId(elementId);
    container.innerHTML = "";

    if (!rows || rows.length === 0) {
        const empty = document.createElement("div");
        empty.className = "table-empty";
        empty.textContent = "No rows";
        container.appendChild(empty);
        updateManualMatchButtonState();
        return;
    }

    const columns = Object.keys(rows[0]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    const selectHeader = document.createElement("th");
    selectHeader.className = "check-cell";
    selectHeader.textContent = "Select";
    headRow.appendChild(selectHeader);

    columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col;
        headRow.appendChild(th);
    });

    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");

        const checkCell = document.createElement("td");
        checkCell.className = "check-cell";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = checkboxName;
        checkbox.value = String(idx);
        checkbox.addEventListener("change", updateManualMatchButtonState);
        checkCell.appendChild(checkbox);
        tr.appendChild(checkCell);

        columns.forEach((col) => {
            const td = document.createElement("td");
            td.textContent = row[col] ?? "";
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
    updateManualMatchButtonState();
}

function renderUnmatchedTransactionsTable() {
    renderSelectableUnmatchedTable(
        "unmatched-transactions-table",
        state.unmatchedTransactions,
        "unmatched-transaction-select",
    );
}

function renderUnmatchedProofsTable() {
    renderSelectableUnmatchedTable(
        "unmatched-proofs-table",
        state.unmatchedProofs,
        "unmatched-proof-select",
    );
}

function manualMatchSelectedUnmatchedRows() {
    const selectedTx = getSelectedIndices("unmatched-transaction-select");
    const selectedProofs = getSelectedIndices("unmatched-proof-select");

    if (!selectedTx.length || !selectedProofs.length) {
        showError("Select at least one unmatched transaction and one unmatched proof.");
        return;
    }

    if (selectedTx.length !== selectedProofs.length) {
        showError("Select the same number of unmatched transactions and unmatched proofs.");
        return;
    }

    showError("");

    const txRows = selectedTx
        .map((idx) => ({ idx, row: state.unmatchedTransactions[idx] }))
        .filter((item) => item.row);
    const proofRows = selectedProofs
        .map((idx) => ({ idx, row: state.unmatchedProofs[idx] }))
        .filter((item) => item.row);

    const pairCount = Math.min(txRows.length, proofRows.length);
    if (pairCount === 0) {
        showError("No valid unmatched rows were selected.");
        return;
    }

    for (let i = 0; i < pairCount; i += 1) {
        const tx = txRows[i].row;
        const proof = proofRows[i].row;
        state.validatedTransactions.push({
            "Transaction Business Name": tx["Business Name"],
            "Transaction Total": tx.Total,
            "Transaction Date": tx.Date,
            "Proof Business Name": proof["Business Name"],
            "Proof Total": proof.Total,
            "Proof Date": proof.Date,
            Reason: "Manually Matched",
        });
    }

    const selectedTxSet = new Set(txRows.slice(0, pairCount).map((item) => item.idx));
    const selectedProofSet = new Set(proofRows.slice(0, pairCount).map((item) => item.idx));

    state.unmatchedTransactions = state.unmatchedTransactions.filter((_, idx) => !selectedTxSet.has(idx));
    state.unmatchedProofs = state.unmatchedProofs.filter((_, idx) => !selectedProofSet.has(idx));

    updateMetrics({
        validatedTransactions: state.validatedTransactions,
        discrepancies: state.discrepancies,
        unmatchedTransactions: state.unmatchedTransactions,
        unmatchedProofs: state.unmatchedProofs,
    });

    renderTable("validated-table", state.validatedTransactions);
    renderUnmatchedTransactionsTable();
    renderUnmatchedProofsTable();
    updateResultPanelsVisibility();

    byId("download-btn").disabled = state.validatedTransactions.length === 0;
    byId("summary-text").textContent = `Manually matched ${pairCount} unmatched pair(s).`;
}

function updateMetrics(payload) {
    void payload;
}

function normalizeCell(value) {
    if (typeof value === "number") {
        return Number(value.toFixed(2));
    }

    const raw = value ?? "";
    const numeric = Number(raw);
    if (!Number.isNaN(numeric) && String(raw).trim() !== "") {
        return Number(numeric.toFixed(2));
    }

    return String(raw).trim().toLowerCase();
}

function removeMatchedRow(rows, target) {
    return rows.filter((row) => {
        const sameName = normalizeCell(row["Business Name"]) === normalizeCell(target.name);
        const sameTotal = normalizeCell(row.Total) === normalizeCell(target.total);
        const sameDate = normalizeCell(row.Date) === normalizeCell(target.date);
        return !(sameName && sameTotal && sameDate);
    });
}

function acceptSelectedRecommendations() {
    const selected = Array.from(
        document.querySelectorAll("input[name='recommendation-select']:checked"),
    ).map((node) => Number(node.value));

    if (!selected.length) {
        showError("Select at least one recommendation to accept.");
        return;
    }

    showError("");
    const selectedRows = selected.map((idx) => state.recommendations[idx]).filter(Boolean);

    selectedRows.forEach((rec) => {
        state.validatedTransactions.push({
            "Transaction Business Name": rec["Transaction Business Name"],
            "Transaction Total": rec["Transaction Total"],
            "Transaction Date": rec["Transaction Date"],
            "Proof Business Name": rec["Proof Business Name"],
            "Proof Total": rec["Proof Total"],
            "Proof Date": rec["Proof Date"],
            Result: "Recommended",
        });

        state.unmatchedTransactions = removeMatchedRow(state.unmatchedTransactions, {
            name: rec["Transaction Business Name"],
            total: rec["Transaction Total"],
            date: rec["Transaction Date"],
        });

        state.unmatchedProofs = removeMatchedRow(state.unmatchedProofs, {
            name: rec["Proof Business Name"],
            total: rec["Proof Total"],
            date: rec["Proof Date"],
        });
    });

    state.recommendations = state.recommendations.filter((_, idx) => !selected.includes(idx));

    updateMetrics({
        validatedTransactions: state.validatedTransactions,
        discrepancies: state.discrepancies,
        unmatchedTransactions: state.unmatchedTransactions,
        unmatchedProofs: state.unmatchedProofs,
    });

    renderTable("validated-table", state.validatedTransactions);
    renderUnmatchedTransactionsTable();
    renderUnmatchedProofsTable();
    renderRecommendationsTable();
    updateResultPanelsVisibility();

    byId("download-btn").disabled = state.validatedTransactions.length === 0;
    byId("summary-text").textContent = `Accepted ${selectedRows.length} recommendation(s).`;
}

function validateSelectedDiscrepancies() {
    const selected = Array.from(
        document.querySelectorAll("input[name='discrepancy-select']:checked"),
    ).map((node) => Number(node.value));

    if (!selected.length) {
        showError("Select at least one discrepancy row to validate.");
        return;
    }

    const selectedRows = selected.map((idx) => state.discrepancies[idx]).filter(Boolean);
    if (!selectedRows.length) {
        showError("No valid discrepancy rows selected.");
        return;
    }

    showError("");

    selectedRows.forEach((row, position) => {
        const idx = selected[position];
        const amountInput = byId(`adjusted-amount-${idx}`);
        const commentInput = byId(`discrepancy-comment-${idx}`);

        const adjustedAmount = Number.parseFloat(amountInput?.value ?? "");
        const finalAmount = Number.isNaN(adjustedAmount)
            ? row["Transaction Total"]
            : Number(adjustedAmount.toFixed(2));

        const comment = (commentInput?.value ?? "").trim();

        const validatedRow = {
            "Transaction Business Name": row["Transaction Business Name"],
            "Transaction Total": finalAmount,
            "Transaction Date": row["Transaction Date"],
            "Proof Business Name": row["Proof Business Name"],
            "Proof Total": row["Proof Total"],
            "Proof Date": row["Proof Date"],
            Result: "Validated (Manual)",
        };

        if (comment) {
            validatedRow.Comment = comment;
        }

        state.validatedTransactions.push(validatedRow);
    });

    state.discrepancies = state.discrepancies.filter((_, idx) => !selected.includes(idx));

    updateMetrics({
        validatedTransactions: state.validatedTransactions,
        discrepancies: state.discrepancies,
        unmatchedTransactions: state.unmatchedTransactions,
        unmatchedProofs: state.unmatchedProofs,
    });

    renderTable("validated-table", state.validatedTransactions);
    renderDiscrepanciesTable();
    updateResultPanelsVisibility();

    byId("download-btn").disabled = state.validatedTransactions.length === 0;
    byId("summary-text").textContent = `Moved ${selectedRows.length} discrepancy row(s) into validated records.`;
}

function clearAll({ keepSessionIds = false } = {}) {
    byId("transactions-input").value = "";
    byId("proofs-input").value = "";
    if (!keepSessionIds) {
        byId("session-id").value = "";
        byId("load-session-id").value = "";
    }
    byId("summary-text").textContent = "No validation has been executed yet.";
    showError("");
    setStatus("Idle");

    const emptyPayload = {
        validatedTransactions: [],
        discrepancies: [],
        unmatchedTransactions: [],
        unmatchedProofs: [],
    };

    state.validatedTransactions = [];
    state.discrepancies = [];
    state.unmatchedTransactions = [];
    state.unmatchedProofs = [];
    state.recommendations = [];
    state.loadedTransactions = [];
    state.loadedProofs = [];
    state.chatHistory = [];

    byId("download-btn").disabled = true;
    byId("accept-recommendations-btn").disabled = true;
    byId("validate-discrepancies-btn").disabled = true;
    byId("manual-match-btn").disabled = true;
    resetProgress();

    updateMetrics(emptyPayload);
    dataSourcePanelState.transactionsCollapsed = true;
    dataSourcePanelState.proofsCollapsed = true;
    renderDataSourceTables();
    renderTable("validated-table", []);
    renderDiscrepanciesTable();
    renderUnmatchedTransactionsTable();
    renderUnmatchedProofsTable();
    renderRecommendationsTable();
    renderChatTranscript();
    updateResultPanelsVisibility();
}

async function createSession({ clearUi = true } = {}) {
    if (clearUi) {
        clearAll();
    }
    setStatus("Creating session...");
    showError("");

    try {
        const response = await fetch("/api/session/new", { method: "POST" });
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(payload.error || "Could not create session.");
        }

        byId("session-id").value = payload.sessionId;
        byId("load-session-id").value = payload.sessionId;
        setStatus("Session ready");
        return payload.sessionId;
    } catch (err) {
        const fallbackSessionId = buildLocalSessionId();
        byId("session-id").value = fallbackSessionId;
        byId("load-session-id").value = fallbackSessionId;
        setStatus("Session ready");
        showError(`Session service unavailable. Using local session: ${fallbackSessionId}`);
        return fallbackSessionId;
    }
}

async function saveSession() {
    const sessionId = byId("session-id").value.trim();
    showError("");

    if (!sessionId) {
        showError("Create or load a session before saving.");
        return;
    }

    const snapshot = {
        summary: byId("summary-text").textContent,
        loadedTransactions: state.loadedTransactions,
        loadedProofs: state.loadedProofs,
        validatedTransactions: state.validatedTransactions,
        discrepancies: state.discrepancies,
        unmatchedTransactions: state.unmatchedTransactions,
        unmatchedProofs: state.unmatchedProofs,
        recommendations: state.recommendations,
        chatHistory: state.chatHistory,
    };

    setStatus("Saving session...");

    try {
        const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}/save`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ state: snapshot }),
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || "Could not save session.");
        }

        setStatus("Session saved");
    } catch (err) {
        setStatus("Error");
        showError(err.message);
    }
}

async function runValidation() {
    let sessionId = byId("session-id").value.trim();
    const transactionFiles = byId("transactions-input").files;
    const proofFiles = byId("proofs-input").files;

    showError("");

    if (!sessionId) {
        const createdSessionId = await createSession({ clearUi: false });
        if (!createdSessionId) {
            showError("Could not create a session. Please try again.");
            return;
        }
        sessionId = createdSessionId;
    }

    const hasUploads = transactionFiles.length > 0 || proofFiles.length > 0;
    const hasLoadedInputs = state.loadedTransactions.length > 0 && state.loadedProofs.length > 0;

    if (hasUploads && (!transactionFiles.length || !proofFiles.length)) {
        showError("Upload both transaction and proof files, or upload neither to use saved inputs.");
        return;
    }

    if (!hasUploads && !hasLoadedInputs) {
        showError(
            "Please upload transaction/proof files, or load a session with saved inputs first.",
        );
        return;
    }

    const formData = new FormData();
    formData.append("sessionId", sessionId);
    for (const file of transactionFiles) {
        formData.append("transactions", file);
    }
    for (const file of proofFiles) {
        formData.append("proofs", file);
    }

    isValidationRunning = true;
    setStatus("Validating...");
    byId("summary-text").textContent = hasUploads
        ? "Validation in progress. Parsing files and matching records..."
        : "Validation in progress using saved session inputs...";
    startProgress();

    try {
        const response = await fetch("/api/validate", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Validation failed.");
        }

        state.validatedTransactions = payload.validatedTransactions;
        state.discrepancies = payload.discrepancies;
        state.unmatchedTransactions = payload.unmatchedTransactions;
        state.unmatchedProofs = payload.unmatchedProofs;
        state.recommendations = payload.recommendations;
        state.loadedTransactions = payload.transactions ?? state.loadedTransactions;
        state.loadedProofs = payload.proofs ?? state.loadedProofs;

        byId("download-btn").disabled = payload.validatedTransactions.length === 0;
        byId("summary-text").textContent = payload.summary;
        updateMetrics(payload);

        dataSourcePanelState.transactionsCollapsed = true;
        dataSourcePanelState.proofsCollapsed = true;
        renderDataSourceTables();
        renderTable("validated-table", payload.validatedTransactions);
        renderDiscrepanciesTable();
        renderUnmatchedTransactionsTable();
        renderUnmatchedProofsTable();
        renderRecommendationsTable();
        updateResultPanelsVisibility();

        completeProgress();
        isValidationRunning = false;
        setStatus("Validation Complete");
    } catch (err) {
        isValidationRunning = false;
        resetProgress();
        setStatus("Error");
        showError(err.message);
    }
}

async function loadSessionInputs() {
    const sessionId = byId("load-session-id").value.trim();
    showError("");

    if (!sessionId) {
        showError("Enter a session_id to load previous inputs.");
        return;
    }

    setStatus("Loading session...");

    try {
        const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}`);
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Could not load session.");
        }

        clearAll();
        byId("session-id").value = payload.sessionId;
        byId("load-session-id").value = payload.sessionId;
        state.loadedTransactions = payload.transactions;
        state.loadedProofs = payload.proofs;
        dataSourcePanelState.transactionsCollapsed = true;
        dataSourcePanelState.proofsCollapsed = true;
        renderDataSourceTables();

        const stateResponse = await fetch(
            `/api/session/${encodeURIComponent(payload.sessionId)}/state`,
        );
        const statePayload = await stateResponse.json().catch(() => ({}));
        if (!stateResponse.ok) {
            throw new Error(statePayload.error || "Could not load saved session state.");
        }

        if (statePayload.state && typeof statePayload.state === "object") {
            state.loadedTransactions = statePayload.state.loadedTransactions ?? payload.transactions;
            state.loadedProofs = statePayload.state.loadedProofs ?? payload.proofs;
            state.validatedTransactions = statePayload.state.validatedTransactions ?? [];
            state.discrepancies = statePayload.state.discrepancies ?? [];
            state.unmatchedTransactions = statePayload.state.unmatchedTransactions ?? [];
            state.unmatchedProofs = statePayload.state.unmatchedProofs ?? [];
            state.recommendations = statePayload.state.recommendations ?? [];
            state.chatHistory = statePayload.state.chatHistory ?? [];

            dataSourcePanelState.transactionsCollapsed = true;
            dataSourcePanelState.proofsCollapsed = true;
            renderDataSourceTables();
            renderTable("validated-table", state.validatedTransactions);
            renderDiscrepanciesTable();
            renderUnmatchedTransactionsTable();
            renderUnmatchedProofsTable();
            renderRecommendationsTable();
            renderChatTranscript();

            byId("summary-text").textContent =
                statePayload.state.summary ||
                `Loaded session ${payload.sessionId} with saved progress.`;
            byId("download-btn").disabled = state.validatedTransactions.length === 0;
        } else {
            byId("summary-text").textContent = `Loaded session ${payload.sessionId} with ${payload.transactions.length} transaction rows and ${payload.proofs.length} proof rows.`;
        }

        updateResultPanelsVisibility();
        setStatus("Session loaded");
    } catch (err) {
        setStatus("Error");
        showError(err.message);
    }
}

async function sendChatMessage() {
    const sessionId = byId("session-id")?.value.trim() || "";
    const input = byId("chat-input");
    const sendBtn = byId("chat-send-btn");
    const message = (input?.value || "").trim();

    if (!sessionId) {
        showError("Create or load a session before chatting.");
        return;
    }

    if (!message) {
        return;
    }

    showError("");
    setChatPopupOpen(true);
    state.chatHistory.push({ role: "user", text: message });
    renderChatTranscript();
    input.value = "";

    if (sendBtn) {
        sendBtn.disabled = true;
    }
    if (input) {
        input.disabled = true;
    }

    state.chatIsStreaming = true;
    const assistantIndex = state.chatHistory.length;
    state.chatHistory.push({
        role: "assistant",
        text: randomBufferingPhrase(),
        pending: true,
    });
    renderChatTranscript();

    try {
        const response = await fetch("/api/chat/ask/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ sessionId, message }),
        });
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.error || "Chat request failed.");
        }

        if (!response.body) {
            throw new Error("Streaming response body is unavailable.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamDone = false;

        while (!streamDone) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const blocks = buffer.split("\n\n");
            buffer = blocks.pop() || "";

            for (const block of blocks) {
                const lines = block.split("\n");
                let eventName = "message";
                const dataLines = [];

                for (const line of lines) {
                    if (line.startsWith("event:")) {
                        eventName = line.slice(6).trim();
                    } else if (line.startsWith("data:")) {
                        dataLines.push(line.slice(5).trim());
                    }
                }

                const raw = dataLines.join("\n");
                const payload = raw ? JSON.parse(raw) : {};
                const currentMsg = state.chatHistory[assistantIndex];

                if (eventName === "token") {
                    if (currentMsg) {
                        if (currentMsg.pending) {
                            currentMsg.pending = false;
                            currentMsg.text = "";
                        }
                        currentMsg.text += payload.token || "";
                        renderChatTranscript();
                    }
                } else if (eventName === "done") {
                    if (currentMsg) {
                        currentMsg.pending = false;
                        if (!currentMsg.text.trim()) {
                            currentMsg.text =
                                payload.answer || "I could not generate an answer.";
                        }
                        currentMsg.chart = payload.chart || null;
                        currentMsg.top_categories = payload.top_categories || null;
                        currentMsg.comparison_table = payload.comparison_table || null;
                        renderChatTranscript();
                    }
                    streamDone = true;
                } else if (eventName === "error") {
                    throw new Error(payload.error || "Chat stream failed.");
                }
            }
        }

        const currentMsg = state.chatHistory[assistantIndex];
        if (currentMsg) {
            currentMsg.pending = false;
            if (!currentMsg.text.trim()) {
                currentMsg.text = "I could not generate an answer.";
            }
            renderChatTranscript();
        }
    } catch (err) {
        const currentMsg = state.chatHistory[assistantIndex];
        if (currentMsg) {
            currentMsg.pending = false;
            currentMsg.text = `Error: ${err.message}`;
        } else {
            state.chatHistory.push({
                role: "assistant",
                text: `Error: ${err.message}`,
            });
        }
        renderChatTranscript();
    } finally {
        state.chatIsStreaming = false;
        if (sendBtn) {
            sendBtn.disabled = false;
        }
        if (input) {
            input.disabled = false;
            input.focus();
        }
    }
}

async function downloadValidatedPdf() {
    if (!state.validatedTransactions.length) {
        return;
    }

    setStatus("Preparing PDF...");

    try {
        const response = await fetch("/api/export/validated", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ rows: state.validatedTransactions }),
        });

        if (!response.ok) {
            const payload = await response.json();
            throw new Error(payload.error || "PDF export failed.");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "validated_transactions.pdf";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        setStatus("PDF downloaded");
    } catch (err) {
        setStatus("Error");
        showError(err.message);
    }
}

// Signal that main app handlers are active so the inline fallback click logic stays disabled.
window.__arveeAppReady = true;

byId("new-session-btn").addEventListener("click", createSession);
byId("load-session-btn").addEventListener("click", loadSessionInputs);
byId("save-session-btn").addEventListener("click", saveSession);
byId("validate-btn").addEventListener("click", runValidation);
byId("clear-btn").addEventListener("click", clearAll);
byId("download-btn").addEventListener("click", downloadValidatedPdf);
byId("accept-recommendations-btn").addEventListener("click", acceptSelectedRecommendations);
byId("validate-discrepancies-btn").addEventListener("click", validateSelectedDiscrepancies);
byId("manual-match-btn").addEventListener("click", manualMatchSelectedUnmatchedRows);
byId("chat-send-btn").addEventListener("click", sendChatMessage);
byId("chat-fab").addEventListener("click", () => {
    const popup = byId("chat-popup");
    setChatPopupOpen(Boolean(popup?.hidden));
});
byId("chat-back-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    state.chatHistory = [];
    renderChatTranscript();
});
byId("chat-close-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    setChatPopupOpen(false);
});
byId("chat-input").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
        return;
    }
    event.preventDefault();
    sendChatMessage();
});

byId("loaded-transactions-toggle")?.addEventListener("click", () => {
    setDataSourceCollapsed("transactions", !dataSourcePanelState.transactionsCollapsed);
});
byId("loaded-proofs-toggle")?.addEventListener("click", () => {
    setDataSourceCollapsed("proofs", !dataSourcePanelState.proofsCollapsed);
});
byId("add-transactions-btn")?.addEventListener("click", () => {
    byId("transactions-input")?.click();
});
byId("add-proofs-btn")?.addEventListener("click", () => {
    byId("proofs-input")?.click();
});
byId("transactions-input")?.addEventListener("change", () => {
    promptRerunAfterAdd();
});
byId("proofs-input")?.addEventListener("change", () => {
    promptRerunAfterAdd();
});
byId("clear-transactions-btn")?.addEventListener("click", () => {
    state.loadedTransactions = [];
    if (byId("transactions-input")) {
        byId("transactions-input").value = "";
    }
    dataSourcePanelState.transactionsCollapsed = true;
    renderDataSourceTables();
    updateResultPanelsVisibility();
});
byId("clear-proofs-btn")?.addEventListener("click", () => {
    state.loadedProofs = [];
    if (byId("proofs-input")) {
        byId("proofs-input").value = "";
    }
    dataSourcePanelState.proofsCollapsed = true;
    renderDataSourceTables();
    updateResultPanelsVisibility();
});

startGreetingFlash();
setChatPopupOpen(false);

clearAll();

// Auto-load a test session when ?testSession=<id> is in the URL.
(function autoLoadTestSession() {
    const params = new URLSearchParams(window.location.search);
    const testSession = params.get("testSession");
    if (testSession) {
        byId("load-session-id").value = testSession;
        loadSessionInputs();
    }
})();
