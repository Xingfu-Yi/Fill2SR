import test from 'node:test';
import assert from 'node:assert/strict';
import {fitScale, constrain, zoomAt, pinch} from '../docs/geometry.mjs';

test('fit centers a square image without distorting it', () => {
  const scale = fitScale(1000, 600, 1536, 1536);
  assert.deepEqual(constrain({scale,x:0,y:0},1000,600,1536,1536), {scale,x:200,y:0});
});
test('zoom keeps the same image point under the cursor', () => {
  const old = {scale:.5, x:100, y:0};
  const point = {x:400, y:250};
  const next = zoomAt(old,1,point);
  assert.equal((point.x-old.x)/old.scale, (point.x-next.x)/next.scale);
  assert.equal((point.y-old.y)/old.scale, (point.y-next.y)/next.scale);
});
test('panning cannot expose blank margins around an enlarged image', () => {
  assert.deepEqual(constrain({scale:1,x:50,y:-2000},800,600,1536,1536), {scale:1,x:0,y:-936});
});
test('pinch doubles scale and follows the moving midpoint', () => {
  assert.deepEqual(pinch({scale:1,x:0,y:0},[{x:100,y:100},{x:200,y:100}],[{x:100,y:150},{x:300,y:150}],.3,4), {scale:2,x:-100,y:-50});
});
test('pinch respects fit and maximum zoom limits', () => {
  const initial=[{x:100,y:100},{x:200,y:100}];
  assert.equal(pinch({scale:1,x:0,y:0},initial,[{x:149,y:100},{x:151,y:100}],.4,4).scale,.4);
  assert.equal(pinch({scale:1,x:0,y:0},initial,[{x:-900,y:100},{x:1100,y:100}],.4,4).scale,4);
});
