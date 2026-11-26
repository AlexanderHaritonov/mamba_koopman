// Zoom and pan of the map on a drive page, with svg-pan-zoom.
// Lines keep their width on screen (style.css). Dots keep their size: on every zoom, each dot shape shrinks by the zoom.
// matplotlib draws all dots of a line from one shape in <defs>; a CSS transform on it does not reach the dots in Chrome.
// Track labels keep their size too: each label shrinks by the zoom around the track end.
// The axes and the grid are not in the map SVG: they are drawn here again on every pan and zoom.
var SVG_NAMESPACE = "http://www.w3.org/2000/svg";
var PIXELS_BETWEEN_TICKS = 80;  // smallest distance between two ticks on screen
var TICK_LENGTH = 5;
var START_VIEW_METRES = 60;  // at the start the shorter side of the map shows this much

var mapElement = document.getElementById("map");
var map = mapElement.querySelector("svg");
var xMin = Number(mapElement.dataset.xMin);
var xMax = Number(mapElement.dataset.xMax);
var yMax = Number(mapElement.dataset.yMax);
var hasEgo = "egoX" in mapElement.dataset;  // no ego position: drives without samples
var egoX = Number(mapElement.dataset.egoX);
var egoY = Number(mapElement.dataset.egoY);
var xAxis = document.getElementById("x-axis");
var yAxis = document.getElementById("y-axis");
var grid = document.getElementById("map-grid");
var dotShapes = map.querySelectorAll('g[id^="axes"] > g[id^="line2d"] > defs > path');
var panZoom = null;

// matplotlib draws each label as glyph shapes in a group with transform="translate(x y) scale(s -s)".
// (x, y) is the track end, labelOffset points to the right and up (SVG y points down).
var labelOffset = Number(mapElement.dataset.labelOffset);
var labelGroups = map.querySelectorAll('g[id^="text_"] > g');
var labelEndsX = [];
var labelEndsY = [];
var labelScales = [];
for (var i = 0; i < labelGroups.length; i++) {
    var transforms = labelGroups[i].transform.baseVal;
    var translate = transforms.getItem(0).matrix;
    var scale = transforms.getItem(1).matrix;
    labelEndsX.push(translate.e - labelOffset);
    labelEndsY.push(translate.f + labelOffset);
    labelScales.push(scale.a);
}

// Dots and labels the same size on screen at every zoom.
function keepMarkSizes(zoom) {
    for (var i = 0; i < dotShapes.length; i++) {
        dotShapes[i].setAttribute("transform", "scale(" + 1 / zoom + ")");
    }
    for (var j = 0; j < labelGroups.length; j++) {
        labelGroups[j].setAttribute("transform",
            "translate(" + labelEndsX[j] + " " + labelEndsY[j] + ") scale(" + 1 / zoom + ") " +
            "translate(" + labelOffset + " " + -labelOffset + ") scale(" + labelScales[j] + " " + -labelScales[j] + ")");
    }
}

// The smallest step of 1, 2 or 5 times a power of ten that is at least minimumStep.
function tickStep(minimumStep) {
    var power = Math.pow(10, Math.floor(Math.log10(minimumStep)));
    var factors = [1, 2, 5, 10];
    for (var i = 0; i < factors.length; i++) {
        if (factors[i] * power >= minimumStep) {
            return factors[i] * power;
        }
    }
    return 10 * power;
}

// All multiples of step between low and high.
function tickValues(low, high, step) {
    var values = [];
    for (var k = Math.ceil(low / step); k * step <= high; k++) {
        values.push(k * step);
    }
    return values;
}

function tickText(value, step) {
    var decimals = Math.max(0, -Math.floor(Math.log10(step)));
    var text = value.toFixed(decimals);
    if (Number(text) === 0) {
        text = (0).toFixed(decimals);  // no "-0"
    }
    return text;
}

function addLine(parent, className, x1, y1, x2, y2) {
    var line = document.createElementNS(SVG_NAMESPACE, "line");
    line.setAttribute("class", className);
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    parent.appendChild(line);
}

function addText(parent, x, y, anchor, text) {
    var label = document.createElementNS(SVG_NAMESPACE, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", y);
    label.setAttribute("text-anchor", anchor);
    label.setAttribute("dominant-baseline", "middle");
    label.textContent = text;
    parent.appendChild(label);
}

// Screen pixels per metre at the current zoom.
function currentPixelsPerMetre() {
    var sizes = panZoom.getSizes();
    // the map SVG fills its axes, so its viewBox width is xMax - xMin metres
    return sizes.realZoom * sizes.viewBox.width / (xMax - xMin);
}

// The ego car in the middle, START_VIEW_METRES on the shorter side.
// Without an ego position, or if the whole drive is smaller: the whole drive.
function showStartView() {
    panZoom.fit();
    panZoom.center();
    if (!hasEgo) {
        return;
    }
    var sizes = panZoom.getSizes();
    var wantedPixelsPerMetre = Math.min(sizes.width, sizes.height) / START_VIEW_METRES;
    if (wantedPixelsPerMetre <= currentPixelsPerMetre()) {
        return;
    }
    panZoom.zoomBy(wantedPixelsPerMetre / currentPixelsPerMetre());
    var pixelsPerMetre = currentPixelsPerMetre();
    panZoom.pan({ x: sizes.width / 2 - (egoX - xMin) * pixelsPerMetre,
                  y: sizes.height / 2 - (yMax - egoY) * pixelsPerMetre });
}

function drawAxes() {
    var pan = panZoom.getPan();
    var sizes = panZoom.getSizes();
    var pixelsPerMetre = currentPixelsPerMetre();
    var leftMetres = xMin - pan.x / pixelsPerMetre;
    var rightMetres = xMin + (sizes.width - pan.x) / pixelsPerMetre;
    var topMetres = yMax + pan.y / pixelsPerMetre;
    var bottomMetres = yMax - (sizes.height - pan.y) / pixelsPerMetre;
    var step = tickStep(PIXELS_BETWEEN_TICKS / pixelsPerMetre);

    // the axes and the grid are not exactly where the map is: shift by the difference
    var mapBox = map.getBoundingClientRect();
    var xAxisBox = xAxis.getBoundingClientRect();
    var yAxisBox = yAxis.getBoundingClientRect();
    var gridBox = grid.getBoundingClientRect();
    xAxis.replaceChildren();
    yAxis.replaceChildren();
    grid.replaceChildren();

    var xValues = tickValues(leftMetres, rightMetres, step);
    for (var i = 0; i < xValues.length; i++) {
        var x = pan.x + (xValues[i] - xMin) * pixelsPerMetre;
        var xOnAxis = x + mapBox.left - xAxisBox.left;
        var xOnGrid = x + mapBox.left - gridBox.left;
        addLine(xAxis, "tick-line", xOnAxis, 0, xOnAxis, TICK_LENGTH);
        addText(xAxis, xOnAxis, TICK_LENGTH + 8, "middle", tickText(xValues[i], step));
        addLine(grid, "grid-line", xOnGrid, 0, xOnGrid, gridBox.height);
    }
    var yValues = tickValues(bottomMetres, topMetres, step);
    var yAxisWidth = yAxisBox.width;
    for (var j = 0; j < yValues.length; j++) {
        var y = pan.y + (yMax - yValues[j]) * pixelsPerMetre;
        var yOnAxis = y + mapBox.top - yAxisBox.top;
        var yOnGrid = y + mapBox.top - gridBox.top;
        addLine(yAxis, "tick-line", yAxisWidth - TICK_LENGTH, yOnAxis, yAxisWidth, yOnAxis);
        addText(yAxis, yAxisWidth - TICK_LENGTH - 3, yOnAxis, "end", tickText(yValues[j], step));
        addLine(grid, "grid-line", 0, yOnGrid, gridBox.width, yOnGrid);
    }
}

panZoom = svgPanZoom(map, {
    controlIconsEnabled: true,
    fit: true,
    center: true,
    minZoom: 1,
    maxZoom: 200,
    zoomScaleSensitivity: 0.3,
    onZoom: keepMarkSizes,
    onUpdatedCTM: function () {
        if (panZoom !== null) {
            drawAxes();
        }
    }
});
showStartView();
drawAxes();

window.addEventListener("resize", function () {
    panZoom.resize();
    showStartView();
    drawAxes();
});
