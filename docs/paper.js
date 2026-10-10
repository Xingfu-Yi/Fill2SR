// Enlarge complete paper figures without hiding the comparison gallery.
const dialog = document.getElementById('figure-dialog');
const body = document.getElementById('figure-dialog-body');
const image = document.getElementById('figure-dialog-image');
const controls = {
  width: document.getElementById('figure-width'),
  fit: document.getElementById('figure-fit'),
};
const hints = {
  width: 'Scroll to explore the full figure. Click or tap the image to close.',
  fit: 'The entire figure fits in the window. Click or tap the image to close.',
};
let previousOverflow;
let opener;
let imagePress;
function setView(mode) {
  body.dataset.view = mode;
  for (const [name, control] of Object.entries(controls)) {
    control.classList.toggle('active', name === mode);
    control.setAttribute('aria-pressed', String(name === mode));
  }
  document.getElementById('figure-dialog-note').textContent = hints[mode];
  body.scrollTo(0, 0);
}
for (const button of document.querySelectorAll('.figure-expand')) {
  button.addEventListener('click', () => {
    opener = button;
    imagePress = null;
    const source = button.querySelector('img');
    image.src = source.src;
    image.alt = source.alt;
    document.getElementById('figure-dialog-title').textContent = button.dataset.figureTitle;
    previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.showModal();
    setView('width');
  });
}
for (const [mode, control] of Object.entries(controls)) {
  control.addEventListener('click', () => setView(mode));
}
document.getElementById('figure-close').addEventListener('click', () => dialog.close());
// A click dismisses the image; dragging, scrolling and pinch gestures do not.
image.addEventListener('pointerdown', event => {
  imagePress = event.isPrimary && event.button === 0
    ? {id: event.pointerId, x: event.clientX, y: event.clientY,
       top: body.scrollTop, left: body.scrollLeft, moved: false}
    : null;
});
image.addEventListener('pointermove', event => {
  if (imagePress && imagePress.id === event.pointerId &&
      Math.hypot(event.clientX - imagePress.x, event.clientY - imagePress.y) > 8) {
    imagePress.moved = true;
  }
});
image.addEventListener('pointercancel', () => {imagePress = null;});
image.addEventListener('click', event => {
  const press = imagePress;
  imagePress = null;
  if (event.detail === 0 || (press && !press.moved &&
      body.scrollTop === press.top && body.scrollLeft === press.left)) dialog.close();
});
image.addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    dialog.close();
  }
});
dialog.addEventListener('close', () => {
  imagePress = null;
  document.body.style.overflow = previousOverflow;
  opener?.focus({preventScroll: true});
});
dialog.addEventListener('click', event => {if (event.target === dialog) dialog.close();});
