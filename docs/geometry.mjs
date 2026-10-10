// Shared image coordinates: both layers always receive the same transform.
export const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
export const fitScale = (w, h, iw, ih) => Math.min(w / iw, h / ih);
export function constrain(view, w, h, iw, ih) {
  const sw = iw * view.scale, sh = ih * view.scale;
  return {...view, x: sw <= w ? (w - sw) / 2 : clamp(view.x, w - sw, 0),
    y: sh <= h ? (h - sh) / 2 : clamp(view.y, h - sh, 0)};
}
export function zoomAt(view, scale, point) {
  const ratio = scale / view.scale;
  return {scale, x: point.x - (point.x - view.x) * ratio,
    y: point.y - (point.y - view.y) * ratio};
}
export function pinch(view, oldPoints, newPoints, minScale, maxScale) {
  const middle = p => ({x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2});
  const distance = p => Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  const before = middle(oldPoints), after = middle(newPoints);
  const scale = clamp(view.scale * distance(newPoints) / Math.max(1, distance(oldPoints)), minScale, maxScale);
  const next = zoomAt(view, scale, before);
  return {...next, x: next.x + after.x - before.x, y: next.y + after.y - before.y};
}
