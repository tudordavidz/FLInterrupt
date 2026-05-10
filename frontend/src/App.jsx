import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const defaultConfig = {
  num_clients: 5,
  rounds: 6,
  local_epochs: 1,
  samples_per_client: 600,
  batch_size: 64,
  lr: 0.01,
  seed: 42,
  dataset_name: "cifar10",
  data_distribution: "iid",
  model_name: "resnet18",
  transfer_learning: true,
};

const defaultCvConfig = {
  repeats: 2,
  k_folds: 5,
  max_samples: 3000,
};

/** User-adjustable scale for figure text and chart chrome (stored in localStorage). */
const CHART_FONT_SCALE = {
  min: 0.85,
  max: 2.2,
  step: 0.1,
  default: 1.55,
  /** Legacy single-scale key; read once when migrating to `scalesStorageKey`. */
  storageKey: "fl_chart_font_scale",
  scalesStorageKey: "fl_chart_font_scales",
};

function clampChartFontScale(raw) {
  const v = raw == null ? NaN : Number(raw);
  if (!Number.isFinite(v)) {
    return CHART_FONT_SCALE.default;
  }
  const c = Math.min(
    CHART_FONT_SCALE.max,
    Math.max(CHART_FONT_SCALE.min, v)
  );
  return Math.round(c * 100) / 100;
}

const LEGEND_NUDGE_KEYS = {
  evolution: "fl_chart_legend_nudge_evolution",
  cv: "fl_chart_legend_nudge_cv",
};

function loadLegendNudge(key) {
  try {
    const raw = localStorage.getItem(key);
    if (raw) {
      const o = JSON.parse(raw);
      if (Number.isFinite(o.dx) && Number.isFinite(o.dy)) {
        return { dx: o.dx, dy: o.dy };
      }
    }
  } catch {
    /* ignore */
  }
  return { dx: 0, dy: 0 };
}

function clientPointToCanvas(canvas, clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / Math.max(r.width, 1e-6);
  const sy = canvas.height / Math.max(r.height, 1e-6);
  return { x: (clientX - r.left) * sx, y: (clientY - r.top) * sy };
}

function pointInLegendRect(x, y, rect) {
  if (!rect) {
    return false;
  }
  return (
    x >= rect.left &&
    x <= rect.left + rect.width &&
    y >= rect.top &&
    y <= rect.top + rect.height
  );
}

function loadChartFontScales() {
  try {
    const raw = localStorage.getItem(CHART_FONT_SCALE.scalesStorageKey);
    if (raw) {
      const o = JSON.parse(raw);
      return {
        evolution: clampChartFontScale(o.evolution),
        cv: clampChartFontScale(o.cv),
        confusion: clampChartFontScale(o.confusion),
      };
    }
    const legacy = localStorage.getItem(CHART_FONT_SCALE.storageKey);
    if (legacy != null) {
      const v = clampChartFontScale(parseFloat(legacy));
      return { evolution: v, cv: v, confusion: v };
    }
  } catch {
    /* ignore */
  }
  return {
    evolution: CHART_FONT_SCALE.default,
    cv: CHART_FONT_SCALE.default,
    confusion: CHART_FONT_SCALE.default,
  };
}

function chartPx(scale, n) {
  return Math.max(6, Math.round(n * scale));
}

function makeChartTheme(scaleRaw) {
  const s = Math.min(
    CHART_FONT_SCALE.max,
    Math.max(CHART_FONT_SCALE.min, scaleRaw)
  );
  const p = (n) => chartPx(s, n);
  const legend = {
    markerGap: p(11),
    itemGap: p(26),
    rowGap: p(10),
    rowHeight: p(22),
    padX: p(14),
    padY: p(10),
    titleHeight: p(20),
    titleGap: p(8),
    lineW: p(24),
    dotR: p(3),
    sq: p(14),
    sqHalf: p(7),
    markerStroke: Math.max(1, 2.5 * s),
  };
  return {
    scale: s,
    tickFont: `800 ${p(18)}px JetBrains Mono, monospace`,
    axisTitleFont: `600 ${p(18)}px Plus Jakarta Sans, system-ui, sans-serif`,
    axisCaptionFont: `600 ${p(18)}px Plus Jakarta Sans, system-ui, sans-serif`,
    emptyStateFont: `500 ${p(22)}px Plus Jakarta Sans, system-ui, sans-serif`,
    legendTitleFont: `700 ${p(17)}px Plus Jakarta Sans, system-ui, sans-serif`,
    legendItemFont: `600 ${p(17)}px Plus Jakarta Sans, system-ui, sans-serif`,
    statusMonoFont: `${p(15)}px JetBrains Mono, monospace`,
    cmNoteFont: `${p(14)}px JetBrains Mono, monospace`,
    cmErrFont: `500 ${p(18)}px Plus Jakarta Sans, system-ui, sans-serif`,
    plotLeftGutter: p(120),
    plotRightGutter: p(94),
    plotPadX: p(50),
    plotPadY: p(26),
    tickPadOuterL: p(12),
    tickPadOuterR: p(14),
    yAxisLabelOffset: p(68),
    seriesLineWidth: Math.max(1.5, 2.2 * s),
    dualAxisLineWidth: Math.max(1.5, 2 * s),
    gridLineWidth: 1,
    pointRadius: Math.max(3, 3.6 * s),
    pointStrokeWidth: Math.max(1, 1.4 * s),
    legendPlotGap: p(26),
    evolution: {
      statusRowPitch: p(22),
      statusBottomPad: p(52),
      /** Must exceed `axisCaptionDy` so “Rounds” sits above the client matrix with a small gap. */
      plotBottomGap: p(66),
      clipTopPad: p(28),
      roundLabelDy: p(22),
      axisCaptionDy: p(50),
      emptyStateX: p(24),
      emptyStateY: p(44),
      statusCell: p(14),
    },
    cv: {
      bottomReserve: p(100),
      foldLabelDy: p(26),
      captionDy: p(56),
    },
    legend,
  };
}

const LABEL_MAP = {
  num_clients: "Number of Clients",
  rounds: "Rounds",
  local_epochs: "Local Epochs",
  samples_per_client: "Target Samples per Client",
  batch_size: "Batch Size",
  lr: "Learning Rate",
  seed: "Seed",
  dataset_name: "Dataset",
  data_distribution: "Data Distribution",
  model_name: "Model",
  transfer_learning: "Transfer Learning",
  repeats: "Repeats",
  k_folds: "K Folds",
  max_samples: "Max Samples",
};

function prettyLabel(key) {
  return LABEL_MAP[key] || key.replaceAll("_", " ");
}

/**
 * Canvas size for confusion matrix: heatmap side = n * minCellPx (large n → chunky cells).
 * Height > width to fit axis titles + horizontal color bar without shrinking the grid.
 */
function confusionMatrixCanvasDimensions(n, scale = CHART_FONT_SCALE.default) {
  const p = (x) => chartPx(scale, x);
  const cap = 8192;
  if (n === 0) {
    return { w: p(960), h: p(960) };
  }

  let minCell;
  if (n <= 12) {
    minCell = Math.max(10, Math.round((880 - 260) / n));
  } else if (n <= 20) {
    minCell = Math.max(9, Math.round((1040 - 300) / n));
  } else if (n <= 30) {
    minCell = Math.max(8, Math.round((1280 - 340) / n));
  } else if (n <= 50) {
    minCell = Math.max(7, Math.round((1680 - 380) / n));
  } else if (n <= 100) {
    minCell = n > 90 ? 16 : n > 70 ? 17 : 18;
  } else {
    minCell = 14;
  }

  const size = Math.ceil(n * minCell);
  // Left pad must fit y-axis tick labels + gap + vertical "True class" (see drawConfusionMatrix).
  const padL = n <= 12 ? p(210) : n <= 20 ? p(162) : n <= 50 ? p(112) : p(88);
  const padR = p(52);
  // Top margin for horizontal class names (no diagonal labels).
  const padT = n <= 20 ? p(108) : p(100);
  const padB = p(168);

  return {
    w: Math.min(cap, size + padL + padR),
    h: Math.min(cap, size + padT + padB),
  };
}

function downloadCanvas(canvas, filename) {
  if (!canvas) {
    return;
  }
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function drawLineSeries(ctx, points, color, theme) {
  if (points.length < 2) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = theme.seriesLineWidth;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
}

function drawPointSeries(ctx, points, color, theme) {
  if (!points.length) {
    return;
  }

  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
  ctx.lineWidth = theme.pointStrokeWidth;

  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, theme.pointRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

/** Acc axis: 0, 0.1, …, 1.0;Loss axis shares the same grid (scaled by lossMax). */
function formatAccTickLabel(t) {
  if (t <= 0) {
    return "0";
  }
  if (t >= 1) {
    return "1";
  }
  return (Math.round(t * 10) / 10).toFixed(1);
}

function drawLegendMarkerLine(ctx, x, y, color, theme) {
  const { lineW, dotR, markerStroke } = theme.legend;
  const mid = lineW / 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = markerStroke;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + lineW, y);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x + mid, y, dotR, 0, Math.PI * 2);
  ctx.fill();
}

function drawLegendMarkerSquare(ctx, x, y, color, theme) {
  const { sq, sqHalf } = theme.legend;
  ctx.fillStyle = color;
  ctx.fillRect(x, y - sqHalf, sq, sq);
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y - sqHalf, sq, sq);
}

function computeSeriesLegendLayout(ctx, items, theme, maxRowPackWidth) {
  const {
    markerGap,
    itemGap,
    rowGap,
    rowHeight,
    padX,
    padY,
    titleHeight,
    titleGap,
  } = theme.legend;
  const lineMarkerW = theme.legend.lineW;
  const squareMarkerW = theme.legend.sq;
  const maxRowWidth = Math.max(80, maxRowPackWidth);

  ctx.font = theme.legendItemFont;
  const rows = [[]];
  let currentRowWidth = 0;

  for (const item of items) {
    const markerWidth = item.marker === "line" ? lineMarkerW : squareMarkerW;
    const itemWidth = markerWidth + markerGap + ctx.measureText(item.label).width;
    const nextWidth = rows[rows.length - 1].length === 0
      ? itemWidth
      : currentRowWidth + itemGap + itemWidth;

    if (nextWidth > maxRowWidth && rows[rows.length - 1].length > 0) {
      rows.push([]);
      currentRowWidth = 0;
    }

    rows[rows.length - 1].push({ ...item, markerWidth, itemWidth });
    currentRowWidth = rows[rows.length - 1].length === 1
      ? itemWidth
      : currentRowWidth + itemGap + itemWidth;
  }

  const boxHeight =
    padY * 2 +
    titleHeight +
    titleGap +
    rows.length * rowHeight +
    (rows.length - 1) * rowGap;

  ctx.font = theme.legendTitleFont;
  const titleTextW = ctx.measureText("Legend").width;
  ctx.font = theme.legendItemFont;

  let maxContent = titleTextW;
  for (const row of rows) {
    let rw = 0;
    row.forEach((item, idx) => {
      rw += (idx ? itemGap : 0) + item.itemWidth;
    });
    maxContent = Math.max(maxContent, rw);
  }
  const boxWidth = maxContent + padX * 2;

  return { rows, boxWidth, boxHeight };
}

function paintSeriesLegend(ctx, theme, layout, boxLeft, boxTop) {
  const {
    rows,
    boxWidth,
    boxHeight,
  } = layout;
  const {
    markerGap,
    itemGap,
    rowGap,
    rowHeight,
    padX,
    padY,
    titleHeight,
    titleGap,
  } = theme.legend;

  ctx.fillStyle = "rgba(255,255,255,0.94)";
  ctx.fillRect(boxLeft, boxTop, boxWidth, boxHeight);
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(boxLeft, boxTop, boxWidth, boxHeight);

  ctx.font = theme.legendTitleFont;
  ctx.fillStyle = "#0f172a";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Legend", boxLeft + padX, boxTop + padY + titleHeight - 2);

  ctx.font = theme.legendItemFont;
  ctx.fillStyle = "#334155";
  ctx.textBaseline = "middle";

  rows.forEach((row, rowIdx) => {
    let x = boxLeft + padX;
    const y =
      boxTop +
      padY +
      titleHeight +
      titleGap +
      rowIdx * (rowHeight + rowGap) +
      rowHeight / 2;

    row.forEach((item) => {
      if (item.marker === "line") {
        drawLegendMarkerLine(ctx, x, y, item.color, theme);
      } else {
        drawLegendMarkerSquare(ctx, x, y, item.color, theme);
      }

      const textX = x + item.markerWidth + markerGap;
      ctx.fillStyle = "#334155";
      ctx.fillText(item.label, textX, y);
      x = textX + ctx.measureText(item.label).width + itemGap;
    });
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

/** Layout uses this as max row width; large value keeps every item on one horizontal row. */
const LEGEND_SINGLE_ROW_PACK_WIDTH = 1e6;

/**
 * Upper-right inside the plot frame by default; `legendNudge` shifts in canvas pixels (clamped).
 * Returns pixel rect for hit-testing / drag.
 */
function drawInsetSeriesLegend(ctx, items, theme, plotLeft, plotTop, plotRight, plotBottom, legendNudge) {
  const dx = legendNudge?.dx ?? 0;
  const dy = legendNudge?.dy ?? 0;
  const p = (n) => chartPx(theme.scale, n);
  const margin = p(12);
  const plotW = plotRight - plotLeft;
  const plotH = plotBottom - plotTop;
  if (plotW < 80 || plotH < 60) {
    return null;
  }
  const layout = computeSeriesLegendLayout(ctx, items, theme, LEGEND_SINGLE_ROW_PACK_WIDTH);
  const boxWidth = layout.boxWidth;
  const boxHeight = layout.boxHeight;

  let boxLeft = plotRight - margin - boxWidth + dx;
  let boxTop = plotTop + margin + dy;
  boxLeft = Math.max(plotLeft + margin, Math.min(boxLeft, plotRight - margin - boxWidth));
  boxTop = Math.max(plotTop + margin, Math.min(boxTop, plotBottom - margin - boxHeight));
  paintSeriesLegend(ctx, theme, { ...layout, boxWidth }, boxLeft, boxTop);
  return { left: boxLeft, top: boxTop, width: boxWidth, height: boxHeight };
}

function ChartFontToolbar({ ariaLabel, scale, onChange }) {
  return (
    <div className="chart-font-toolbar chart-font-toolbar--embedded" role="group" aria-label={ariaLabel}>
      <span className="chart-font-toolbar__label">Figure text</span>
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        onClick={() =>
          onChange(
            Math.max(
              CHART_FONT_SCALE.min,
              Math.round((scale - CHART_FONT_SCALE.step) * 10) / 10
            )
          )
        }
        disabled={scale <= CHART_FONT_SCALE.min}
      >
        Smaller
      </button>
      <span className="chart-font-toolbar__value" aria-live="polite">
        {Math.round(scale * 100)}%
      </span>
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        onClick={() =>
          onChange(
            Math.min(
              CHART_FONT_SCALE.max,
              Math.round((scale + CHART_FONT_SCALE.step) * 10) / 10
            )
          )
        }
        disabled={scale >= CHART_FONT_SCALE.max}
      >
        Larger
      </button>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => onChange(CHART_FONT_SCALE.default)}>
        Reset
      </button>
    </div>
  );
}

function drawDualAxisTicks(ctx, left, right, top, bottom, lossMax, theme) {
  ctx.font = theme.tickFont;
  ctx.lineCap = "butt";

  for (let i = 0; i <= 10; i += 1) {
    const t = i / 10;
    const y = bottom - t * (bottom - top);
    const isBase = t === 0;
    const isMajor = i % 5 === 0;
    ctx.strokeStyle = isBase ? "#cbd5e1" : isMajor ? "#e2e8f0" : "#f1f5f9";
    ctx.lineWidth = isBase ? 1.25 : 1;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();

    const labelY = isBase ? bottom - 4 : y + 5;
    const lossVal = t * lossMax;
    const lossStr = lossVal.toFixed(2);

    ctx.fillStyle = "#2563eb";
    ctx.textAlign = "right";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(formatAccTickLabel(t), left - theme.tickPadOuterL, labelY);
    ctx.textAlign = "left";

    ctx.fillStyle = "#dc2626";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(lossStr, right + theme.tickPadOuterR, labelY);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function drawDualAxes(ctx, left, right, top, bottom, theme) {
  const lw = theme.dualAxisLineWidth;
  ctx.lineCap = "butt";
  ctx.lineWidth = lw;
  ctx.strokeStyle = "#2563eb";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.stroke();

  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = theme.gridLineWidth;
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.stroke();
}

function drawDualYAxisTitles(ctx, left, right, top, bottom, theme) {
  ctx.font = theme.axisTitleFont;
  const midY = (top + bottom) / 2;

  ctx.save();
  ctx.translate(left - theme.yAxisLabelOffset, midY);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#2563eb";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Acc", 0, 0);
  ctx.restore();

  ctx.save();
  ctx.translate(right + theme.yAxisLabelOffset, midY);
  ctx.rotate(Math.PI / 2);
  ctx.fillStyle = "#dc2626";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Loss", 0, 0);
  ctx.restore();

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function drawGlobalEvolution(canvas, history, theme, legendNudge = { dx: 0, dy: 0 }) {
  if (!canvas) {
    return { legendRect: null };
  }

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const ev = theme.evolution;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!history.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = theme.emptyStateFont;
    ctx.fillText("No rounds completed yet", ev.emptyStateX, ev.emptyStateY);
    return { legendRect: null };
  }

  const left = theme.plotLeftGutter;
  const right = width - theme.plotRightGutter;
  const legend = [
    { label: "train acc", color: "#059669", marker: "line" },
    { label: "val acc", color: "#2563eb", marker: "line" },
    { label: "train loss", color: "#d97706", marker: "line" },
    { label: "val loss", color: "#dc2626", marker: "line" },
    { label: "connected", color: "#059669", marker: "square" },
    { label: "disconnected", color: "#dc2626", marker: "square" },
  ];
  const top = chartPx(theme.scale, 14);
  const rounds = history.map((h) => Number(h.round));
  const maxRound = Math.max(...rounds);
  const clientIds = Array.from(
    new Set(
      history.flatMap((h) => (h.client_submission || []).map((s) => Number(s.client_id)))
    )
  ).sort((a, b) => a - b);

  const statusRowPitch = ev.statusRowPitch;
  const statusTop = height - ev.statusBottomPad - Math.max(1, clientIds.length) * statusRowPitch;
  const bottom = statusTop - ev.plotBottomGap;
  const lossMax = Math.max(
    1,
    ...history.flatMap((h) => [h.train_loss ?? 0, h.val_loss ?? 0])
  );

  const plotPadX = theme.plotPadX;
  const plotLeft = left + plotPadX;
  const plotRight = right - plotPadX;

  const xForRound = (r) => {
    const rn = Number(r);
    return plotLeft + ((rn - 1) / Math.max(maxRound - 1, 1)) * (plotRight - plotLeft);
  };

  const yAcc = (v) => bottom - v * (bottom - top);
  const yLoss = (v) => bottom - (v / lossMax) * (bottom - top);

  const trainAccPoints = history.map((h) => ({ x: xForRound(h.round), y: yAcc(h.train_acc ?? 0) }));
  const valAccPoints = history.map((h) => ({ x: xForRound(h.round), y: yAcc(h.val_acc ?? 0) }));
  const trainLossPoints = history.map((h) => ({ x: xForRound(h.round), y: yLoss(h.train_loss ?? 0) }));
  const valLossPoints = history.map((h) => ({ x: xForRound(h.round), y: yLoss(h.val_loss ?? 0) }));

  const clipTop = top - ev.clipTopPad;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, clipTop, width, bottom - clipTop + 2);
  ctx.clip();

  drawDualAxisTicks(ctx, left, right, top, bottom, lossMax, theme);
  drawDualAxes(ctx, left, right, top, bottom, theme);
  drawDualYAxisTitles(ctx, left, right, top, bottom, theme);

  // Vertical split guides for each round.
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = theme.gridLineWidth;
  for (const h of history) {
    const x = xForRound(h.round);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
  }

  drawLineSeries(ctx, trainAccPoints, "#059669", theme);
  drawLineSeries(ctx, valAccPoints, "#2563eb", theme);
  drawLineSeries(ctx, trainLossPoints, "#d97706", theme);
  drawLineSeries(ctx, valLossPoints, "#dc2626", theme);
  drawPointSeries(ctx, trainAccPoints, "#059669", theme);
  drawPointSeries(ctx, valAccPoints, "#2563eb", theme);
  drawPointSeries(ctx, trainLossPoints, "#d97706", theme);
  drawPointSeries(ctx, valLossPoints, "#dc2626", theme);

  ctx.restore();

  const legendRect = drawInsetSeriesLegend(
    ctx,
    legend,
    theme,
    plotLeft,
    top,
    plotRight,
    bottom,
    legendNudge
  );

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, statusTop, width, height - statusTop);

  ctx.font = theme.statusMonoFont;
  const cell = ev.statusCell;
  for (const h of history) {
    const x = xForRound(h.round);
    ctx.fillStyle = "#64748b";
    ctx.fillText(String(h.round), x - 3, bottom + ev.roundLabelDy);

    const subMap = new Map((h.client_submission || []).map((s) => [Number(s.client_id), !!s.connected]));
    ctx.textAlign = "right";
    ctx.textBaseline = "alphabetic";
    clientIds.forEach((cid, rowIdx) => {
      const yCell = statusTop + rowIdx * statusRowPitch;
      const v = subMap.get(cid);
      ctx.fillStyle = "#475569";
      ctx.fillText(`C${cid}`, x - 9, yCell + Math.round(statusRowPitch / 2) + 3);
      ctx.fillStyle = v == null ? "#e2e8f0" : v ? "#059669" : "#dc2626";
      ctx.fillRect(x - 6, yCell, cell, cell);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 6, yCell, cell, cell);
    });
    ctx.textAlign = "left";
  }

  ctx.fillStyle = "#334155";
  ctx.font = theme.axisCaptionFont;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Rounds", (plotLeft + plotRight) / 2, bottom + ev.axisCaptionDy);
  ctx.textAlign = "left";

  return { legendRect };
}

function drawCrossValidation(canvas, foldResults, theme, legendNudge = { dx: 0, dy: 0 }) {
  if (!canvas) {
    return { legendRect: null };
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const cv = theme.cv;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!foldResults?.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = theme.emptyStateFont;
    ctx.fillText("Run repeated K-fold validation to render this graph", theme.evolution.emptyStateX, theme.evolution.emptyStateY);
    return { legendRect: null };
  }

  const left = theme.plotLeftGutter;
  const right = width - theme.plotRightGutter;
  const legend = [
    { label: "Train Accuracy", color: "#059669", marker: "line" },
    { label: "Val Accuracy", color: "#2563eb", marker: "line" },
    { label: "Train Loss", color: "#d97706", marker: "line" },
    { label: "Val Loss", color: "#dc2626", marker: "line" },
  ];
  const top = chartPx(theme.scale, 12);
  const bottom = height - cv.bottomReserve;
  const plotPadX = theme.plotPadX;
  const plotPadY = theme.plotPadY;
  const plotLeft = left + plotPadX;
  const plotRight = right - plotPadX;
  const plotTop = top + plotPadY;
  const plotBottom = bottom - plotPadY;

  const count = foldResults.length;
  const maxLoss = Math.max(
    1,
    ...foldResults.map((f) =>
      Math.max(
        Number(f.train_loss ?? f.loss ?? 0),
        Number(f.val_loss ?? f.loss ?? 0)
      )
    )
  );
  drawDualAxisTicks(ctx, left, right, plotTop, plotBottom, maxLoss, theme);
  drawDualAxes(ctx, left, right, plotTop, plotBottom, theme);
  drawDualYAxisTitles(ctx, left, right, plotTop, plotBottom, theme);
  const xFor = (i) =>
    plotLeft + (i / Math.max(count - 1, 1)) * (plotRight - plotLeft);

  // Vertical split guides for each fold position.
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = theme.gridLineWidth;
  for (let i = 0; i < count; i += 1) {
    const x = xFor(i);
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotBottom);
    ctx.stroke();
  }

  const yAcc = (v) => plotBottom - v * (plotBottom - plotTop);
  const yLoss = (v) => plotBottom - (v / maxLoss) * (plotBottom - plotTop);

  const trainAccPts = foldResults.map((f, i) => ({
    x: xFor(i),
    y: yAcc(Number(f.train_accuracy ?? f.accuracy ?? 0)),
  }));
  const valAccPts = foldResults.map((f, i) => ({
    x: xFor(i),
    y: yAcc(Number(f.val_accuracy ?? f.accuracy ?? 0)),
  }));
  const trainLossPts = foldResults.map((f, i) => ({
    x: xFor(i),
    y: yLoss(Number(f.train_loss ?? f.loss ?? 0)),
  }));
  const valLossPts = foldResults.map((f, i) => ({
    x: xFor(i),
    y: yLoss(Number(f.val_loss ?? f.loss ?? 0)),
  }));

  drawLineSeries(ctx, trainAccPts, "#059669", theme);
  drawLineSeries(ctx, valAccPts, "#2563eb", theme);
  drawLineSeries(ctx, trainLossPts, "#d97706", theme);
  drawLineSeries(ctx, valLossPts, "#dc2626", theme);
  drawPointSeries(ctx, trainAccPts, "#059669", theme);
  drawPointSeries(ctx, valAccPts, "#2563eb", theme);
  drawPointSeries(ctx, trainLossPts, "#d97706", theme);
  drawPointSeries(ctx, valLossPts, "#dc2626", theme);

  const legendRect = drawInsetSeriesLegend(
    ctx,
    legend,
    theme,
    plotLeft,
    plotTop,
    plotRight,
    plotBottom,
    legendNudge
  );

  ctx.fillStyle = "#64748b";
  ctx.font = theme.statusMonoFont;
  ctx.textAlign = "center";
  for (let i = 0; i < count; i += 1) {
    const foldLabel = foldResults[i]?.fold ?? i + 1;
    ctx.fillText(String(foldLabel), xFor(i), plotBottom + cv.foldLabelDy);
  }

  ctx.fillStyle = "#334155";
  ctx.font = theme.axisCaptionFont;
  ctx.textAlign = "center";
  ctx.fillText("Fold index", (plotLeft + plotRight) / 2, plotBottom + cv.captionDy);
  ctx.textAlign = "left";

  return { legendRect };
}

/** Class names on confusion matrix axes (e.g. cat, truck); bold + mono for readability. */
function confusionMatrixCategoryFont(axisFontPx) {
  return `800 ${axisFontPx}px JetBrains Mono, monospace`;
}

/** Horizontal column headers: true if centered labels fit between neighbor column centers. */
function confusionMatrixTopLabelsFitHorizontally(ctx, axisFontPx, n, labels, showTick, cell) {
  ctx.font = confusionMatrixCategoryFont(axisFontPx);
  const shown = [];
  for (let i = 0; i < n; i += 1) {
    if (!showTick(i)) {
      continue;
    }
    const label = String(labels?.[i] ?? i).slice(0, n > 60 ? 3 : 12);
    shown.push({ i, w: ctx.measureText(label).width });
  }
  for (let k = 0; k < shown.length - 1; k += 1) {
    const a = shown[k];
    const b = shown[k + 1];
    const dist = (b.i - a.i) * cell;
    if ((a.w + b.w) / 2 > dist * 0.94) {
      return false;
    }
  }
  return true;
}

function drawConfusionMatrix(canvas, matrix, labels, theme) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const p = (n) => chartPx(theme.scale, n);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!matrix?.length) {
    ctx.fillStyle = "#000000";
    ctx.font = theme.emptyStateFont;
    ctx.fillText("Mean confusion matrix will appear after validation", theme.evolution.emptyStateX, theme.evolution.emptyStateY);
    return;
  }

  const n = matrix.length;
  if (matrix.some((row) => row.length !== n)) {
    ctx.fillStyle = "#b91c1c";
    ctx.font = theme.cmErrFont;
    ctx.fillText("Confusion matrix is not square; check API data.", theme.evolution.emptyStateX, theme.evolution.emptyStateY);
    return;
  }

  const tickEvery = n <= 12 ? 1 : n <= 24 ? 2 : n <= 50 ? 5 : n <= 100 ? 10 : 20;
  const showTick = (i) => i === 0 || i === n - 1 || i % tickEvery === 0;
  const showCellValues = n <= 20;
  /** Start size; reduced in the layout loop until horizontal top labels fit (reference-style row). */
  let axisFontPx = n > 80 ? p(17) : n > 40 ? p(20) : n > 20 ? p(21) : p(21);
  const titleFontPx = n > 40 ? p(15) : p(14);

  const yTickToHeatmapGap = p(13);
  const yTitleGap = p(18);

  ctx.font = `500 ${titleFontPx}px Plus Jakarta Sans, system-ui, sans-serif`;
  const trueClassRotatedHSpan = Math.ceil(titleFontPx * 1.35);

  const padR = p(48);
  const padB = p(162);
  const basePadT = showCellValues ? p(72) : p(80);
  const tickYOffset = p(16);
  const minAxisPx = 7;

  let padL = p(58);
  let size = 0;
  let cell = 0;
  let x0 = padL;
  let y0 = basePadT;

  for (let iter = 0; iter < 24; iter += 1) {
    ctx.font = confusionMatrixCategoryFont(axisFontPx);
    let maxYTickW = 0;
    for (let i = 0; i < n; i += 1) {
      if (!showTick(i)) {
        continue;
      }
      const label = String(labels?.[i] ?? i).slice(0, n > 60 ? 3 : 12);
      maxYTickW = Math.max(maxYTickW, ctx.measureText(label).width);
    }

    padL = Math.max(
      p(58),
      maxYTickW + yTickToHeatmapGap + yTitleGap + trueClassRotatedHSpan + p(12)
    );
    size = Math.min(width - padL - padR, height - basePadT - padB);
    if (size < 40) {
      ctx.fillStyle = "#000000";
      ctx.font = theme.cmErrFont;
      ctx.fillText("Canvas too small for this matrix; export PNG for full resolution.", theme.evolution.emptyStateX, theme.evolution.emptyStateY);
      return;
    }
    cell = size / n;

    if (confusionMatrixTopLabelsFitHorizontally(ctx, axisFontPx, n, labels, showTick, cell)) {
      x0 = padL;
      y0 = basePadT;
      break;
    }
    if (axisFontPx <= minAxisPx) {
      x0 = padL;
      y0 = basePadT;
      break;
    }
    axisFontPx -= 1;
  }

  ctx.font = confusionMatrixCategoryFont(axisFontPx);
  let maxYTickW = 0;
  for (let i = 0; i < n; i += 1) {
    if (!showTick(i)) {
      continue;
    }
    const label = String(labels?.[i] ?? i).slice(0, n > 60 ? 3 : 12);
    maxYTickW = Math.max(maxYTickW, ctx.measureText(label).width);
  }
  padL = Math.max(
    p(58),
    maxYTickW + yTickToHeatmapGap + yTitleGap + trueClassRotatedHSpan + p(12)
  );
  size = Math.min(width - padL - padR, height - basePadT - padB);
  if (size < 40) {
    ctx.fillStyle = "#000000";
    ctx.font = theme.cmErrFont;
    ctx.fillText("Canvas too small for this matrix; export PNG for full resolution.", theme.evolution.emptyStateX, theme.evolution.emptyStateY);
    return;
  }
  cell = size / n;
  x0 = padL;
  y0 = basePadT;

  const maxVal = Math.max(...matrix.flat(), 1e-6);
  const showCellBorders = cell >= 3 || n <= 32;
  const subtleBorder = cell >= 1.2 && cell < 3;
  const heatmapNoteY = p(42);
  const predictTitleDy = p(34);
  const barH = p(20);
  const barGap = p(50);

  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const val = matrix[r][c];
      const alpha = Math.min(1, val / maxVal);
      ctx.fillStyle = `rgba(37,99,235,${0.18 + alpha * 0.78})`;
      ctx.fillRect(x0 + c * cell, y0 + r * cell, cell, cell);
      if (showCellBorders) {
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + c * cell, y0 + r * cell, cell, cell);
      } else if (subtleBorder) {
        ctx.strokeStyle = "rgba(226,232,240,0.35)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x0 + c * cell, y0 + r * cell, cell, cell);
      }

      if (showCellValues) {
        const cellFontMax = n <= 12 ? 0.26 : 0.3;
        ctx.font = `${Math.max(p(9), Math.min(p(19), Math.floor(cell * cellFontMax)))}px JetBrains Mono, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = alpha > 0.5 ? "#ffffff" : "#000000";
        ctx.fillText(
          Number(val).toFixed(1),
          x0 + c * cell + cell / 2,
          y0 + r * cell + cell / 2
        );
      }
    }
  }

  ctx.fillStyle = "#000000";
  ctx.font = confusionMatrixCategoryFont(axisFontPx);
  ctx.textBaseline = "middle";

  for (let i = 0; i < n; i += 1) {
    if (!showTick(i)) {
      continue;
    }
    const label = String(labels?.[i] ?? i).slice(0, n > 60 ? 3 : 12);
    const cx = x0 + i * cell + cell / 2;
    ctx.textAlign = "center";
    ctx.fillText(label, cx, y0 - tickYOffset);
    ctx.textAlign = "right";
    ctx.fillText(label, x0 - yTickToHeatmapGap, y0 + i * cell + cell / 2);
  }

  if (!showCellValues) {
    ctx.fillStyle = "#000000";
    ctx.font = theme.cmNoteFont;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(
      `Heatmap only (${n}×${n} classes); values omitted for readability — use Export PNG to inspect pixels.`,
      x0,
      y0 - heatmapNoteY
    );
  }

  ctx.fillStyle = "#000000";
  ctx.font = `500 ${titleFontPx}px Plus Jakarta Sans, system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Predicted class", x0 + size / 2, y0 + size + predictTitleDy);
  ctx.save();
  const yLabelRightX = x0 - yTickToHeatmapGap;
  const trueClassCenterX = yLabelRightX - maxYTickW - yTitleGap - trueClassRotatedHSpan / 2;
  ctx.translate(trueClassCenterX, y0 + size / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("True class", 0, 0);
  ctx.restore();

  const barY = y0 + size + barGap;
  const grad = ctx.createLinearGradient(x0, 0, x0 + size, 0);
  grad.addColorStop(0, "rgba(37,99,235,0.18)");
  grad.addColorStop(1, "rgba(37,99,235,0.96)");
  ctx.fillStyle = grad;
  ctx.fillRect(x0, barY, size, barH);
  ctx.strokeStyle = "#cbd5e1";
  ctx.strokeRect(x0, barY, size, barH);
  ctx.fillStyle = "#000000";
  ctx.font = theme.axisTitleFont;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText("low", x0, barY + barH / 2);
  ctx.textAlign = "right";
  ctx.fillText("high", x0 + size, barY + barH / 2);
}

export default function App() {
  const [cfg, setCfg] = useState(defaultConfig);
  const [cvCfg, setCvCfg] = useState(defaultCvConfig);
  const [state, setState] = useState({
    running: false,
    current_round: 0,
    history: [],
    clients: [],
    logs: [],
    device: "-",
    options: {
      datasets: ["cifar10"],
      distributions: ["iid", "noniid"],
      models: ["resnet18"],
    },
  });
  const [error, setError] = useState("");
  const [cvData, setCvData] = useState(null);
  const [cvLoading, setCvLoading] = useState(false);
  const [stopping, setStopping] = useState(false);

  const [chartFontScales, setChartFontScales] = useState(loadChartFontScales);

  const [evolutionLegendNudge, setEvolutionLegendNudge] = useState(() =>
    loadLegendNudge(LEGEND_NUDGE_KEYS.evolution)
  );
  const [cvLegendNudge, setCvLegendNudge] = useState(() => loadLegendNudge(LEGEND_NUDGE_KEYS.cv));

  const evolutionLegendBoundsRef = useRef(null);
  const cvLegendBoundsRef = useRef(null);
  const legendDragRef = useRef({
    active: false,
    chart: null,
    pointerId: null,
    startCanvasX: 0,
    startCanvasY: 0,
    originDx: 0,
    originDy: 0,
  });

  const evolutionTheme = useMemo(
    () => makeChartTheme(chartFontScales.evolution),
    [chartFontScales.evolution]
  );
  const cvTheme = useMemo(() => makeChartTheme(chartFontScales.cv), [chartFontScales.cv]);
  const confusionTheme = useMemo(
    () => makeChartTheme(chartFontScales.confusion),
    [chartFontScales.confusion]
  );

  useEffect(() => {
    try {
      localStorage.setItem(CHART_FONT_SCALE.scalesStorageKey, JSON.stringify(chartFontScales));
    } catch {
      /* ignore */
    }
  }, [chartFontScales]);

  useEffect(() => {
    try {
      localStorage.setItem(LEGEND_NUDGE_KEYS.evolution, JSON.stringify(evolutionLegendNudge));
    } catch {
      /* ignore */
    }
  }, [evolutionLegendNudge]);

  useEffect(() => {
    try {
      localStorage.setItem(LEGEND_NUDGE_KEYS.cv, JSON.stringify(cvLegendNudge));
    } catch {
      /* ignore */
    }
  }, [cvLegendNudge]);

  const evolutionCanvasId = "evolution-canvas";
  const cvCanvasId = "cv-canvas";
  const cmCanvasId = "cm-canvas";

  async function refresh() {
    try {
      const res = await fetch(`${API_BASE}/api/state`);
      const data = await res.json();
      setState(data);
    } catch {
      setError("Cannot reach backend. Start FastAPI first.");
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, []);

  async function start() {
    setError("");
    setStopping(false);
    try {
      const res = await fetch(`${API_BASE}/api/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Could not start simulation");
      }
      refresh();
    } catch (e) {
      setError(e.message);
    }
  }

  async function interruptClient(clientId) {
    await fetch(`${API_BASE}/api/interrupt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_ids: [clientId] }),
    });
    refresh();
  }

  async function reconnectClient(clientId) {
    await fetch(`${API_BASE}/api/reconnect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_ids: [clientId] }),
    });
    refresh();
  }

  async function stopSimulation() {
    setError("");
    setStopping(true);
    try {
      await fetch(`${API_BASE}/api/stop`, {
        method: "POST",
      });
      refresh();
    } catch {
      setStopping(false);
      setError("Could not stop simulation.");
    }
  }

  useEffect(() => {
    if (!state.running) {
      setStopping(false);
    }
  }, [state.running]);

  async function runCrossValidation() {
    setError("");
    setCvLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/cross-validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...cvCfg,
          dataset_name: cfg.dataset_name,
          training_model_name: cfg.model_name,
          transfer_learning: cfg.transfer_learning,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        const d = data.detail;
        const msg = Array.isArray(d) ? d.map((x) => x.msg || x).join(" ") : d;
        throw new Error(msg || "Cross-validation failed");
      }
      setCvData(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setCvLoading(false);
    }
  }

  const latest = state.history[state.history.length - 1];
  const liveActiveClients = state.clients.filter((c) => c.connected).length;
  const nCm = cvData?.mean_confusion_matrix?.length ?? 0;
  const cmDim = useMemo(
    () => confusionMatrixCanvasDimensions(nCm, chartFontScales.confusion),
    [nCm, chartFontScales.confusion]
  );

  useEffect(() => {
    const el = document.getElementById(evolutionCanvasId);
    const { legendRect } = drawGlobalEvolution(
      el,
      state.history,
      evolutionTheme,
      evolutionLegendNudge
    );
    evolutionLegendBoundsRef.current = legendRect;
  }, [state.history, evolutionTheme, evolutionLegendNudge]);

  useEffect(() => {
    const el = document.getElementById(cvCanvasId);
    const { legendRect } = drawCrossValidation(
      el,
      cvData?.fold_results || [],
      cvTheme,
      cvLegendNudge
    );
    cvLegendBoundsRef.current = legendRect;
  }, [cvData, cvTheme, cvLegendNudge]);

  useEffect(() => {
    drawConfusionMatrix(
      document.getElementById(cmCanvasId),
      cvData?.mean_confusion_matrix || [],
      cvData?.labels || [],
      confusionTheme
    );
  }, [cvData, confusionTheme, cmDim.w, cmDim.h]);

  function onEvolutionLegendPointerDown(e) {
    const canvas = e.currentTarget;
    if (!state.history.length) {
      return;
    }
    const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
    if (!pointInLegendRect(x, y, evolutionLegendBoundsRef.current)) {
      return;
    }
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    legendDragRef.current = {
      active: true,
      chart: "evolution",
      pointerId: e.pointerId,
      startCanvasX: x,
      startCanvasY: y,
      originDx: evolutionLegendNudge.dx,
      originDy: evolutionLegendNudge.dy,
    };
    canvas.style.cursor = "grabbing";
  }

  function onEvolutionLegendPointerMove(e) {
    const canvas = e.currentTarget;
    const d = legendDragRef.current;
    if (d.active && d.chart === "evolution" && d.pointerId === e.pointerId) {
      const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
      setEvolutionLegendNudge({
        dx: d.originDx + (x - d.startCanvasX),
        dy: d.originDy + (y - d.startCanvasY),
      });
      return;
    }
    if (!d.active && state.history.length) {
      const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
      canvas.style.cursor = pointInLegendRect(x, y, evolutionLegendBoundsRef.current) ? "grab" : "";
    }
  }

  function onEvolutionLegendPointerUp(e) {
    const canvas = e.currentTarget;
    const d = legendDragRef.current;
    if (d.active && d.chart === "evolution" && d.pointerId === e.pointerId) {
      legendDragRef.current = {
        active: false,
        chart: null,
        pointerId: null,
        startCanvasX: 0,
        startCanvasY: 0,
        originDx: 0,
        originDy: 0,
      };
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    }
    canvas.style.cursor = "";
  }

  function onEvolutionLegendPointerLeave(e) {
    const d = legendDragRef.current;
    if (d.active && d.chart === "evolution") {
      return;
    }
    e.currentTarget.style.cursor = "";
  }

  function onCvLegendPointerDown(e) {
    const canvas = e.currentTarget;
    if (!cvData?.fold_results?.length) {
      return;
    }
    const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
    if (!pointInLegendRect(x, y, cvLegendBoundsRef.current)) {
      return;
    }
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    legendDragRef.current = {
      active: true,
      chart: "cv",
      pointerId: e.pointerId,
      startCanvasX: x,
      startCanvasY: y,
      originDx: cvLegendNudge.dx,
      originDy: cvLegendNudge.dy,
    };
    canvas.style.cursor = "grabbing";
  }

  function onCvLegendPointerMove(e) {
    const canvas = e.currentTarget;
    const d = legendDragRef.current;
    if (d.active && d.chart === "cv" && d.pointerId === e.pointerId) {
      const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
      setCvLegendNudge({
        dx: d.originDx + (x - d.startCanvasX),
        dy: d.originDy + (y - d.startCanvasY),
      });
      return;
    }
    if (!d.active && cvData?.fold_results?.length) {
      const { x, y } = clientPointToCanvas(canvas, e.clientX, e.clientY);
      canvas.style.cursor = pointInLegendRect(x, y, cvLegendBoundsRef.current) ? "grab" : "";
    }
  }

  function onCvLegendPointerUp(e) {
    const canvas = e.currentTarget;
    const d = legendDragRef.current;
    if (d.active && d.chart === "cv" && d.pointerId === e.pointerId) {
      legendDragRef.current = {
        active: false,
        chart: null,
        pointerId: null,
        startCanvasX: 0,
        startCanvasY: 0,
        originDx: 0,
        originDy: 0,
      };
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    }
    canvas.style.cursor = "";
  }

  function onCvLegendPointerLeave(e) {
    const d = legendDragRef.current;
    if (d.active && d.chart === "cv") {
      return;
    }
    e.currentTarget.style.cursor = "";
  }

  const statusLabel = stopping ? "Stopping" : state.running ? "Running" : "Idle";
  const isConfigLocked = state.running || stopping;

  return (
    <div className="layout">
      <header className="hero">
        <div className="hero-top">
          <p className="eyebrow">Federated learning lab</p>
          <span className={`status-pill status-pill--${stopping ? "stopping" : state.running ? "live" : "idle"}`}>
            <span className="status-pill__dot" aria-hidden />
            {statusLabel}
          </span>
        </div>
        <h1>FL Interrupt Simulator</h1>
      </header>

      <div className="dashboard-grid">
      <section className="card controls">
        <div className="section-head">
          <h2>Simulation</h2>
          <p className="section-desc">Configure rounds, data split, and optimizer — then start the server-side run.</p>
        </div>
        <div className="grid">
          {[
            "num_clients",
            "rounds",
            "local_epochs",
            "samples_per_client",
            "batch_size",
            "lr",
            "seed",
          ].map((key) => (
            <label key={key}>
              <span>{prettyLabel(key)}</span>
              <input
                type="number"
                step={key === "lr" ? "0.001" : "1"}
                value={cfg[key]}
                disabled={isConfigLocked}
                onChange={(e) =>
                  setCfg((old) => {
                    const parsed = Number(String(e.target.value).replace(",", "."));
                    return {
                      ...old,
                      [key]: Number.isFinite(parsed) ? parsed : old[key],
                    };
                  })
                }
              />
            </label>
          ))}

          <label>
            <span>{prettyLabel("dataset_name")}</span>
            <select
              value={cfg.dataset_name}
              disabled={isConfigLocked}
              onChange={(e) => setCfg((old) => ({ ...old, dataset_name: e.target.value }))}
            >
              {(state.options?.datasets || []).map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </label>

          <label>
            <span>{prettyLabel("data_distribution")}</span>
            <select
              value={cfg.data_distribution}
              disabled={isConfigLocked}
              onChange={(e) => setCfg((old) => ({ ...old, data_distribution: e.target.value }))}
            >
              {(state.options?.distributions || []).map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </label>

          <label>
            <span>{prettyLabel("model_name")}</span>
            <select
              value={cfg.model_name}
              disabled={isConfigLocked}
              onChange={(e) => setCfg((old) => ({ ...old, model_name: e.target.value }))}
            >
              {(state.options?.models || []).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>

          <label className="field-checkbox">
            <span>{prettyLabel("transfer_learning")}</span>
            <input
              type="checkbox"
              checked={cfg.transfer_learning}
              disabled={isConfigLocked}
              onChange={(e) => setCfg((old) => ({ ...old, transfer_learning: e.target.checked }))}
            />
          </label>
        </div>
        <div className="buttons">
          <button type="button" className="btn btn-primary" onClick={start} disabled={isConfigLocked}>
            Start run
          </button>
          <button
            type="button"
            className={`btn btn-danger ${stopping ? "is-loading" : ""}`}
            onClick={stopSimulation}
            disabled={!state.running || stopping}
          >
            {stopping ? "Stopping…" : "Stop"}
          </button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      <section className="card metrics">
        <div className="section-head section-head--compact">
          <h2>Latest round</h2>
          <p className="section-desc">Values reflect the most recent completed aggregation step.</p>
        </div>
        <div className="metric-row">
          <article className="metric">
            <span>Train accuracy</span>
            <strong>{latest ? latest.train_acc.toFixed(4) : "—"}</strong>
          </article>
          <article className="metric">
            <span>Train loss</span>
            <strong>{latest ? latest.train_loss.toFixed(4) : "—"}</strong>
          </article>
          <article className="metric">
            <span>Val accuracy</span>
            <strong>{latest ? latest.val_acc.toFixed(4) : "—"}</strong>
          </article>
          <article className="metric">
            <span>Val loss</span>
            <strong>{latest ? latest.val_loss.toFixed(4) : "—"}</strong>
          </article>
          <article className="metric">
            <span>Round</span>
            <strong>{state.current_round}</strong>
          </article>
          <article className="metric">
            <span>Active clients</span>
            <strong>{latest ? latest.active_clients : liveActiveClients}</strong>
          </article>
          <article className="metric metric--wide">
            <span>Device</span>
            <strong className="metric-mono">{state.device}</strong>
          </article>
        </div>
      </section>
      </div>

      <section className="card graph-card">
        <div className="graph-toolbar">
          <div className="graph-toolbar__title">
            <h2>Global model evolution</h2>
            <p className="section-desc graph-legend-hint">
              Drag the legend inside the plot to reposition it (saved in this browser).
            </p>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => downloadCanvas(document.getElementById(evolutionCanvasId), "global-evolution.png")}
          >
            Export PNG
          </button>
        </div>
        <ChartFontToolbar
          ariaLabel="Figure text size for global model evolution"
          scale={chartFontScales.evolution}
          onChange={(next) =>
            setChartFontScales((prev) => ({ ...prev, evolution: clampChartFontScale(next) }))
          }
        />
        <canvas
          id={evolutionCanvasId}
          width="2200"
          height="980"
          className="graph-canvas graph-canvas--legend-drag"
          style={{ touchAction: "none" }}
          role="img"
          aria-label="Global model evolution. Drag the legend box to move it."
          onPointerDown={onEvolutionLegendPointerDown}
          onPointerMove={onEvolutionLegendPointerMove}
          onPointerUp={onEvolutionLegendPointerUp}
          onPointerCancel={onEvolutionLegendPointerUp}
          onPointerLeave={onEvolutionLegendPointerLeave}
        />
      </section>

      <section className="card graph-card">
        <div className="graph-toolbar graph-toolbar--stack">
          <div className="graph-toolbar__title">
            <h2>Repeated K-fold validation</h2>
            <p className="section-desc graph-legend-hint">
              Drag the legend inside the plot to reposition it (saved in this browser).
            </p>
          </div>
          <div className="inline-controls">
            <label>
              <span>{prettyLabel("repeats")}</span>
              <input
                type="number"
                value={cvCfg.repeats}
                min="1"
                onChange={(e) => setCvCfg((old) => ({ ...old, repeats: Number(e.target.value) || old.repeats }))}
              />
            </label>
            <label>
              <span>{prettyLabel("k_folds")}</span>
              <input
                type="number"
                value={cvCfg.k_folds}
                min="2"
                onChange={(e) => setCvCfg((old) => ({ ...old, k_folds: Number(e.target.value) || old.k_folds }))}
              />
            </label>
            <label>
              <span>{prettyLabel("max_samples")}</span>
              <input
                type="number"
                value={cvCfg.max_samples}
                min="500"
                onChange={(e) => setCvCfg((old) => ({ ...old, max_samples: Number(e.target.value) || old.max_samples }))}
              />
            </label>
            <button
              type="button"
              className={`btn btn-primary ${cvLoading ? "is-loading" : ""}`}
              onClick={runCrossValidation}
              disabled={cvLoading || state.running}
            >
              {cvLoading ? "Running…" : "Run validation"}
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => downloadCanvas(document.getElementById(cvCanvasId), "repeated-kfold.png")}
            >
              Export PNG
            </button>
          </div>
        </div>
        <ChartFontToolbar
          ariaLabel="Figure text size for repeated K-fold validation chart"
          scale={chartFontScales.cv}
          onChange={(next) => setChartFontScales((prev) => ({ ...prev, cv: clampChartFontScale(next) }))}
        />
        <canvas
          id={cvCanvasId}
          width="2200"
          height="880"
          className="graph-canvas graph-canvas--legend-drag"
          style={{ touchAction: "none" }}
          role="img"
          aria-label="Repeated K-fold validation chart. Drag the legend box to move it."
          onPointerDown={onCvLegendPointerDown}
          onPointerMove={onCvLegendPointerMove}
          onPointerUp={onCvLegendPointerUp}
          onPointerCancel={onCvLegendPointerUp}
          onPointerLeave={onCvLegendPointerLeave}
        />
      </section>

      <section className="card graph-card graph-card--confusion">
        <div className="graph-toolbar">
          <div className="graph-toolbar__title">
            <h2>Mean confusion matrix</h2>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => downloadCanvas(document.getElementById(cmCanvasId), "mean-confusion-matrix.png")}
          >
            Export PNG
          </button>
        </div>
        <ChartFontToolbar
          ariaLabel="Figure text size for mean confusion matrix"
          scale={chartFontScales.confusion}
          onChange={(next) =>
            setChartFontScales((prev) => ({ ...prev, confusion: clampChartFontScale(next) }))
          }
        />
        <div className="cm-matrix-scroll">
          <canvas
            id={cmCanvasId}
            width={cmDim.w}
            height={cmDim.h}
            className="graph-canvas graph-canvas--native-size"
            style={{ width: `${cmDim.w}px`, height: `${cmDim.h}px`, maxWidth: "none" }}
          />
        </div>
      </section>

      <section className="card clients-section">
        <div className="section-head">
          <h2>Clients</h2>
        </div>
        <div className="client-grid">
          {state.clients.map((c) => (
            <article
              key={c.client_id}
              className={`client-card ${c.connected ? "client-card--up" : c.reconnect_round > state.current_round && state.running ? "client-card--pending" : "client-card--down"}`}
            >
              {(() => {
                const reconnectBlocked =
                  !c.connected &&
                  state.running &&
                  c.reconnect_round > state.current_round;
                const pendingReconnect = reconnectBlocked;

                return (
                  <>
                    <header className="client-card__head">
                      <h3>Client {c.client_id}</h3>
                      <span
                        className={`client-badge ${
                          c.connected
                            ? "client-badge--on"
                            : pendingReconnect
                              ? "client-badge--pending"
                              : "client-badge--off"
                        }`}
                      >
                        {c.connected ? "Online" : pendingReconnect ? "Pending reconnect" : "Offline"}
                      </span>
                    </header>
                    <dl className="client-stats">
                      <div>
                        <dt>Reconnect round</dt>
                        <dd>
                          {c.reconnect_round > 0
                            ? c.reconnect_round
                            : c.connected
                              ? "—"
                              : "manual"}
                        </dd>
                      </div>
                      {pendingReconnect && (
                        <div>
                          <dt>Status</dt>
                          <dd>Waiting for round {c.reconnect_round}</dd>
                        </div>
                      )}
                      {c.last_loss != null && (
                        <div>
                          <dt>Last loss</dt>
                          <dd>{c.last_loss.toFixed(4)}</dd>
                        </div>
                      )}
                      <div>
                        <dt>Interruptions</dt>
                        <dd>{c.interruption_events}</dd>
                      </div>
                      <div>
                        <dt>Reconnects</dt>
                        <dd>{c.reconnect_events}</dd>
                      </div>
                      <div>
                        <dt>Participated</dt>
                        <dd>{c.rounds_participated} rounds</dd>
                      </div>
                      <div>
                        <dt>Missed</dt>
                        <dd>{c.rounds_missed} rounds</dd>
                      </div>
                      <div>
                        <dt>Samples</dt>
                        <dd>{c.samples}</dd>
                      </div>
                      {c.metrics_round != null && (
                        <div>
                          <dt>Metrics round</dt>
                          <dd>{c.metrics_round}</dd>
                        </div>
                      )}
                      {c.train_acc != null && (
                        <div>
                          <dt>Train acc</dt>
                          <dd>{c.train_acc.toFixed(4)}</dd>
                        </div>
                      )}
                      {c.train_loss != null && (
                        <div>
                          <dt>Train loss</dt>
                          <dd>{c.train_loss.toFixed(4)}</dd>
                        </div>
                      )}
                      {c.val_acc != null && (
                        <div>
                          <dt>Val acc</dt>
                          <dd>{c.val_acc.toFixed(4)}</dd>
                        </div>
                      )}
                      {c.val_loss != null && (
                        <div>
                          <dt>Val loss</dt>
                          <dd>{c.val_loss.toFixed(4)}</dd>
                        </div>
                      )}
                    </dl>
                    {cfg.data_distribution === "noniid" && (c.class_distribution || []).length > 0 && (
                      <p className="client-split">
                        <span className="client-split__label">Class split</span>
                        {(c.class_distribution || [])
                          .slice(0, 5)
                          .map((d) => `${d.class_name}: ${d.count}`)
                          .join(" · ")}
                      </p>
                    )}
                    <div className="client-actions">
                      <button
                        type="button"
                        className="btn btn-warning btn-sm"
                        onClick={() => interruptClient(c.client_id)}
                        disabled={!c.connected}
                      >
                        Interrupt
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => reconnectClient(c.client_id)}
                        disabled={c.connected || reconnectBlocked}
                      >
                        {reconnectBlocked ? "Wait for round" : "Reconnect"}
                      </button>
                    </div>
                  </>
                );
              })()}
            </article>
          ))}
        </div>
      </section>

      <section className="card logs">
        <div className="section-head section-head--compact">
          <h2>Server logs</h2>
          <p className="section-desc">Streamed messages from the simulation worker.</p>
        </div>
        <pre>{(state.logs ?? []).join("\n") || "No logs yet."}</pre>
      </section>
    </div>
  );
}
