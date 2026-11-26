// Zoom and pan of the map on a drive page, with svg-pan-zoom.
// Lines keep their width on screen (style.css). Dots keep their size: on every zoom, each dot shape shrinks by the zoom.
// matplotlib draws all dots of a line from one shape in <defs>; a CSS transform on it does not reach the dots in Chrome.
var map = document.querySelector("#map svg");
var dotShapes = map.querySelectorAll('g[id^="axes"] > g[id^="line2d"] > defs > path');
svgPanZoom(map, {
    controlIconsEnabled: true,
    fit: true,
    center: true,
    minZoom: 1,
    maxZoom: 200,
    zoomScaleSensitivity: 0.3,
    onZoom: function (zoom) {
        for (var i = 0; i < dotShapes.length; i++) {
            dotShapes[i].setAttribute("transform", "scale(" + 1 / zoom + ")");
        }
    }
});
