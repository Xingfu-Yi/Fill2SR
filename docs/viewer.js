import {clamp, fitScale, constrain, zoomAt, pinch} from './geometry.mjs';

const $ = id => document.getElementById(id);
const stage = $('stage'), viewer = $('viewer'), divider = $('divider');
const before = $('input-image'), after = $('output-image');
let current, generation = 0, ready = false;
let view = {scale: 1, x: 0, y: 0}, split = 50, fitted = true;
const pointers = new Map();
let dividerPointer = null;
const dims = () => ({w: stage.clientWidth, h: stage.clientHeight,
  iw: current?.output_size[0] || 1536, ih: current?.output_size[1] || 1536});
const localPoint = e => {const r = stage.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top};};
function render() {
  const {w,h,iw,ih} = dims();
  view = constrain(view,w,h,iw,ih);
  for (const image of [before, after]) {
    image.style.width = `${iw}px`; image.style.height = `${ih}px`;
    image.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  }
  $('before-layer').style.clipPath = `inset(0 ${100 - split}% 0 0)`;
  divider.style.left = `${split}%`;
  divider.setAttribute('aria-valuenow', Math.round(split));
  divider.setAttribute('aria-valuetext', `Input ${Math.round(split)} percent, Fill2SR ${Math.round(100-split)} percent`);
  $('zoom-level').value = `${Math.round(view.scale * 100)}%`;
  $('fit').classList.toggle('active', fitted);
  $('actual').classList.toggle('active', !fitted && Math.abs(view.scale-1)<.001);
}
function fit(resetSplit = false) {
  const {w,h,iw,ih} = dims();
  fitted = true; view = {scale: fitScale(w,h,iw,ih),x:0,y:0};
  if (resetSplit) split = 50;
  pointers.clear(); render();
}
function zoom(scale, point) {
  if (!ready) return;
  const {w,h,iw,ih} = dims();
  const minimum = fitScale(w,h,iw,ih);
  fitted = scale <= minimum;
  view = zoomAt(view, clamp(scale,minimum,4), point || {x:w/2,y:h/2});
  render();
}
function setSplit(value) {split = clamp(value,0,100);render();}
function toggleControls(enabled) {
  for (const el of viewer.querySelectorAll('button')) el.disabled = !enabled;
  divider.setAttribute('aria-disabled', String(!enabled));
  stage.setAttribute('aria-busy', String(!enabled));
}
function loadImage(url) {
  return new Promise((resolve,reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`)); image.src = url;
  });
}
async function selectExample(item) {
  const ticket = ++generation;
  ready = false; toggleControls(false);
  for (const name of ['input','output']) $('download-'+name).removeAttribute('href');
  $('loading').hidden = false; $('loading').textContent = `Loading ${item.title.toLowerCase()}…`;
  for (const el of document.querySelectorAll('.sample')) el.setAttribute('aria-pressed',String(el.dataset.id===item.id));
  try {
    const images = await Promise.all([loadImage(item.input),loadImage(item.output)]);
    if (ticket !== generation) return;
    for (let i=0;i<2;i++) {
      const expected = i===0 ? item.input_size : item.output_size;
      if (images[i].naturalWidth!==expected[0] || images[i].naturalHeight!==expected[1]) throw new Error('Image dimensions do not match the manifest.');
    }
    current = item; before.src = images[0].src; after.src = images[1].src;
    before.alt = `Input: ${item.title}, RealLQ250 ${item.id}.png`;
    after.alt = `Fill2SR (Ours-Full): ${item.title}`;
    $('example-title').textContent = item.title;
    $('example-meta').textContent = `RealLQ250 · ${item.id}.png · Fig. ${item.paper_figure}, row ${item.paper_row}`;
    const size = a => a.map(n=>n.toLocaleString('en-US')).join(' × ');
    $('dimensions').textContent = `${size(item.input_size)} → ${size(item.output_size)} px · 2K · 28 steps · Seed 2026`;
    for (const name of ['input','output']) {
      $('download-'+name).href = item[name];
      $('download-'+name).download = `fill2sr-reallq250-${item.id}-${name}.png`;
    }
    ready = true; fit(true); toggleControls(true); $('loading').hidden = true;
  } catch (error) {
    if (ticket !== generation) return;
    $('loading').textContent = 'This comparison could not be loaded.';
    const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = 'Retry';
    retry.addEventListener('click',()=>selectExample(item)); $('loading').append(retry);
    for (const name of ['input','output']) $('download-'+name).removeAttribute('href');
    console.error(error);
  }
}
function createGallery(items) {
  for (const item of items) {
    const button = document.createElement('button'); button.type='button';button.className='sample';button.dataset.id=item.id;
    button.title = `${item.title} · RealLQ250 ${item.id} · Fig. ${item.paper_figure}, row ${item.paper_row}`;
    button.setAttribute('aria-label',`${item.title}, RealLQ250 ${item.id}`);button.setAttribute('aria-pressed','false');
    const frame=document.createElement('span');frame.className='sample-image';
    const image=document.createElement('img');image.src=item.thumbnail;image.alt='';image.width=160;image.height=160;image.loading='lazy';image.decoding='async';frame.append(image);
    const name=document.createElement('span');name.className='sample-name';name.textContent=item.title;
    const id=document.createElement('span');id.className='sample-id';id.textContent=`${item.id} · Fig. ${item.paper_figure}`;
    button.append(frame,name,id);button.addEventListener('click',()=>selectExample(item));$('gallery').append(button);
  }
}

$('fit').addEventListener('click',()=>fit());
$('reset').addEventListener('click',()=>fit(true));
$('actual').addEventListener('click',()=>zoom(1));
$('zoom-in').addEventListener('click',()=>zoom(view.scale*1.35));
$('zoom-out').addEventListener('click',()=>zoom(view.scale/1.35));
stage.addEventListener('wheel',e=>{
  // Keep normal page scrolling until the user explicitly focuses the viewer.
  if (!ready || !stage.contains(document.activeElement)) return;
  e.preventDefault();zoom(view.scale*Math.exp(-clamp(e.deltaY,-120,120)*.003),localPoint(e));
},{passive:false});
stage.addEventListener('keydown',e=>{
  if (e.target!==stage || !ready) return;
  if (['+','=','-','0','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) e.preventDefault();
  if (e.key==='+' || e.key==='=') zoom(view.scale*1.35);
  else if (e.key==='-') zoom(view.scale/1.35);
  else if (e.key==='0') fit(true);
  else if (e.key.startsWith('Arrow')) {
    const changes={ArrowLeft:[40,0],ArrowRight:[-40,0],ArrowUp:[0,40],ArrowDown:[0,-40]};
    const delta=changes[e.key];if(delta){view.x+=delta[0];view.y+=delta[1];render();}
  }
});
divider.addEventListener('keydown',e=>{
  if (!ready) return;
  const delta=e.shiftKey?10:2;
  const actions={ArrowLeft:split-delta,ArrowDown:split-delta,ArrowRight:split+delta,ArrowUp:split+delta,Home:0,End:100};
  if(e.key in actions){e.preventDefault();e.stopPropagation();setSplit(actions[e.key]);}
});
divider.addEventListener('pointerdown',e=>{
  if (!ready || dividerPointer!==null) return;
  e.stopPropagation();e.preventDefault();divider.focus({preventScroll:true});dividerPointer=e.pointerId;divider.setPointerCapture(e.pointerId);
});
divider.addEventListener('pointermove',e=>{
  if (e.pointerId!==dividerPointer) return;
  e.stopPropagation();setSplit(localPoint(e).x/stage.clientWidth*100);
});
const endDivider=e=>{if(e.pointerId===dividerPointer)dividerPointer=null;};
for(const event of ['pointerup','pointercancel','lostpointercapture']) divider.addEventListener(event,endDivider);
stage.addEventListener('pointerdown',e=>{
  if(!ready || e.target.closest('#divider') || e.button>0) return;
  stage.focus({preventScroll:true});stage.setPointerCapture(e.pointerId);pointers.set(e.pointerId,localPoint(e));stage.classList.add('dragging');
});
stage.addEventListener('pointermove',e=>{
  if(!pointers.has(e.pointerId))return;
  const old=[...pointers.values()];const previous=pointers.get(e.pointerId);const point=localPoint(e);
  pointers.set(e.pointerId,point);
  if(pointers.size===2){const {w,h,iw,ih}=dims();view=pinch(view,old,[...pointers.values()],fitScale(w,h,iw,ih),4);fitted=false;}
  else if(pointers.size===1){view.x+=point.x-previous.x;view.y+=point.y-previous.y;}
  render();
});
const endPointer=e=>{pointers.delete(e.pointerId);if(!pointers.size)stage.classList.remove('dragging');};
for(const event of ['pointerup','pointercancel','lostpointercapture'])stage.addEventListener(event,endPointer);
stage.addEventListener('dblclick', e => {
  if (e.target.closest('#divider')) return;
  const {w,h,iw,ih} = dims();
  zoom(view.scale < 1 ? 1 : fitScale(w,h,iw,ih), localPoint(e));
});
new ResizeObserver(()=>{if(current)fit();}).observe(stage);

function fullscreenUI(){
  const expanded=document.fullscreenElement===viewer || viewer.classList.contains('expanded');
  $('fullscreen').textContent=expanded?'Exit full screen ⛶':'Full screen ⛶';
  $('fullscreen').setAttribute('aria-label',expanded?'Exit fullscreen':'Enter fullscreen');
  document.body.style.overflow=viewer.classList.contains('expanded')?'hidden':'';
}
$('fullscreen').addEventListener('click',async()=>{
  if(document.fullscreenElement){await document.exitFullscreen();}
  else if(viewer.classList.contains('expanded'))viewer.classList.remove('expanded');
  else {
    try {if(!viewer.requestFullscreen)throw new Error('Native fullscreen unavailable');await viewer.requestFullscreen();}
    catch {viewer.classList.add('expanded');}
  }
  fullscreenUI();
});
document.addEventListener('fullscreenchange',fullscreenUI);
document.addEventListener('keydown',e=>{if(e.key==='Escape' && viewer.classList.contains('expanded')){viewer.classList.remove('expanded');fullscreenUI();$('fullscreen').focus();}});
$('copy-citation').addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText($('bibtex').textContent);$('copy-citation').textContent='Copied ✓';$('copy-status').textContent='Citation copied to clipboard.';}
  catch{const range=document.createRange();range.selectNodeContents($('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);$('copy-status').textContent='Citation selected. Use your browser’s copy command.';$('copy-citation').textContent='Text selected';}
  setTimeout(()=>$('copy-citation').textContent='Copy BibTeX',2500);
});
toggleControls(false);
try {
  const response=await fetch('gallery.json');if(!response.ok)throw new Error('Gallery manifest unavailable');
  const manifest=await response.json();if(!manifest.examples?.length)throw new Error('No examples found');
  createGallery(manifest.examples);await selectExample(manifest.examples[0]);
}catch(error){$('loading').textContent='The gallery could not be loaded. Please refresh to try again.';console.error(error);}
