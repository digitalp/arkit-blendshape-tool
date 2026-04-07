import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── State ──
let scene, camera, renderer, controls;
let currentModel = null;
let morphMeshes = [];
let refFile = null;   // null = use built-in reference
let targetFile = null;
let resultBlob = null;
let activeTab = 'arkit';

const ARKIT = [
  'eyeBlinkLeft','eyeLookDownLeft','eyeLookInLeft','eyeLookOutLeft','eyeLookUpLeft',
  'eyeSquintLeft','eyeWideLeft','eyeBlinkRight','eyeLookDownRight','eyeLookInRight',
  'eyeLookOutRight','eyeLookUpRight','eyeSquintRight','eyeWideRight',
  'jawForward','jawLeft','jawRight','jawOpen','mouthClose','mouthFunnel','mouthPucker',
  'mouthLeft','mouthRight','mouthSmileLeft','mouthSmileRight','mouthFrownLeft',
  'mouthFrownRight','mouthDimpleLeft','mouthDimpleRight','mouthStretchLeft',
  'mouthStretchRight','mouthRollLower','mouthRollUpper','mouthShrugLower',
  'mouthShrugUpper','mouthPressLeft','mouthPressRight','mouthLowerDownLeft',
  'mouthLowerDownRight','mouthUpperUpLeft','mouthUpperUpRight',
  'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
  'cheekPuff','cheekSquintLeft','cheekSquintRight','noseSneerLeft','noseSneerRight',
  'tongueOut',
];

const VISEMES = [
  'viseme_sil','viseme_PP','viseme_FF','viseme_TH','viseme_DD','viseme_kk',
  'viseme_CH','viseme_SS','viseme_nn','viseme_RR','viseme_aa','viseme_E',
  'viseme_I','viseme_O','viseme_U',
];

// ── Three.js Setup ──
function initScene() {
  const canvas = document.getElementById('canvas3d');
  const rect = canvas.parentElement.getBoundingClientRect();

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(rect.width, rect.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a1a);

  camera = new THREE.PerspectiveCamera(35, rect.width / rect.height, 0.01, 100);
  camera.position.set(0, 1.5, 2.5);

  controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.2, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(2, 3, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8888ff, 0.4);
  fill.position.set(-2, 1, -1);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xff8888, 0.3);
  rim.position.set(0, 2, -3);
  scene.add(rim);

  const grid = new THREE.GridHelper(4, 20, 0x0f3460, 0x0f3460);
  grid.material.opacity = 0.3;
  grid.material.transparent = true;
  scene.add(grid);

  animate();
  window.addEventListener('resize', onResize);
}

function onResize() {
  const rect = renderer.domElement.parentElement.getBoundingClientRect();
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Model Loading ──
function loadGLB(arrayBuffer, filename) {
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
  morphMeshes = [];

  const loader = new GLTFLoader();
  loader.parse(arrayBuffer, '', (gltf) => {
    currentModel = gltf.scene;
    scene.add(currentModel);

    currentModel.traverse((child) => {
      if (child.isMesh && child.morphTargetInfluences && child.morphTargetInfluences.length > 0) {
        morphMeshes.push(child);
      }
    });

    const box = new THREE.Box3().setFromObject(currentModel);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    controls.target.copy(center);
    camera.position.set(center.x, center.y + size.y * 0.3, center.z + maxDim * 1.8);
    controls.update();

    const totalMorphs = morphMeshes.reduce((s, m) => s + m.morphTargetInfluences.length, 0);
    document.getElementById('viewportInfo').textContent =
      `${filename} · ${totalMorphs} morph targets`;

    buildSliders();
  }, (err) => {
    showStatus(`Failed to load GLB: ${err.message}`, 'error');
  });
}

// ── Sliders ──
function buildSliders() {
  const container = document.getElementById('sliderContainer');

  if (morphMeshes.length === 0) {
    container.innerHTML = '<div class="no-targets">No morph targets found in this model</div>';
    return;
  }

  const allNames = [];
  const nameToMeshIndex = {};

  for (const mesh of morphMeshes) {
    const dict = mesh.morphTargetDictionary || {};
    for (const [name, idx] of Object.entries(dict)) {
      if (!nameToMeshIndex[name]) {
        allNames.push(name);
        nameToMeshIndex[name] = [];
      }
      nameToMeshIndex[name].push({ mesh, index: idx });
    }
  }

  renderSliders(allNames, nameToMeshIndex);
}

function renderSliders(allNames, nameToMeshIndex) {
  const container = document.getElementById('sliderContainer');
  container.innerHTML = '';

  let filtered;
  if (activeTab === 'arkit') {
    filtered = allNames.filter(n => ARKIT.includes(n));
  } else if (activeTab === 'viseme') {
    filtered = allNames.filter(n => VISEMES.includes(n));
  } else {
    filtered = allNames;
  }

  if (filtered.length === 0) {
    container.innerHTML = `<div class="no-targets">No ${activeTab} targets in this model</div>`;
    return;
  }

  const groups = {};
  for (const name of filtered) {
    let group = 'Other';
    if (name.startsWith('eye')) group = 'Eyes';
    else if (name.startsWith('jaw')) group = 'Jaw';
    else if (name.startsWith('mouth')) group = 'Mouth';
    else if (name.startsWith('brow')) group = 'Brows';
    else if (name.startsWith('cheek')) group = 'Cheeks';
    else if (name.startsWith('nose')) group = 'Nose';
    else if (name.startsWith('tongue')) group = 'Tongue';
    else if (name.startsWith('viseme')) group = 'Visemes';
    if (!groups[group]) groups[group] = [];
    groups[group].push(name);
  }

  for (const [groupName, names] of Object.entries(groups)) {
    const groupDiv = document.createElement('div');
    groupDiv.className = 'slider-group';

    const title = document.createElement('div');
    title.className = 'slider-group-title';
    title.textContent = `${groupName} (${names.length})`;
    groupDiv.appendChild(title);

    for (const name of names) {
      const row = document.createElement('div');
      row.className = 'slider-row';

      const label = document.createElement('label');
      label.textContent = name;
      label.title = name;

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '1';
      slider.step = '0.01';
      slider.value = '0';

      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = '0';

      slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        val.textContent = v.toFixed(2);
        for (const { mesh, index } of nameToMeshIndex[name]) {
          mesh.morphTargetInfluences[index] = v;
        }
      });

      row.appendChild(label);
      row.appendChild(slider);
      row.appendChild(val);
      groupDiv.appendChild(row);
    }

    container.appendChild(groupDiv);
  }
}

// ── Upload Handling ──
function setupUploadZone(zoneId, onFile) {
  const zone = document.getElementById(zoneId);
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.glb';
  input.style.display = 'none';
  zone.appendChild(input);

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => { zone.classList.remove('dragover'); });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
    input.value = '';
  });
}

async function inspectFile(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch('/api/inspect', { method: 'POST', body: form });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

function renderMeshInfo(info, elementId) {
  const el = document.getElementById(elementId);
  el.style.display = 'block';

  let html = `<strong>${info.meshCount}</strong> meshes, <strong>${info.nodeCount}</strong> nodes<br>`;

  if (info.skeleton && !info.skeleton.valid) {
    html += `<span class="tag">Missing bones: ${info.skeleton.missing.join(', ')}</span><br>`;
    html += `<span class="hint" style="color:#ffc107">Will be auto-added during transfer</span><br>`;
  }

  if (info.detectedFace) {
    html += `Face mesh: <strong>${info.detectedFace.meshName}</strong><br>`;
  }

  for (const mesh of info.meshes) {
    const verts = mesh.primitives.reduce((s, p) => s + p.vertexCount, 0);
    const morphs = mesh.primitives.reduce((s, p) => s + p.morphTargetCount, 0);
    html += `<span class="tag">${mesh.name}</span> ${verts} verts`;
    if (morphs > 0) {
      html += ` · <span class="tag green">${mesh.arkitCount} ARKit</span>`;
      html += `<span class="tag green">${mesh.visemeCount} Visemes</span>`;
    }
    html += '<br>';
  }

  el.innerHTML = html;
}

// ── Transfer ──
async function doTransfer() {
  if (!targetFile) return;

  const btn = document.getElementById('transferBtn');
  const progress = document.getElementById('progressBar');
  const download = document.getElementById('downloadBtn');

  btn.disabled = true;
  btn.textContent = 'Processing...';
  progress.classList.add('active');
  download.style.display = 'none';
  resultBlob = null;

  const usingBuiltIn = !refFile;
  showStatus(
    usingBuiltIn
      ? 'Using built-in reference (brunette.glb)... This may take a moment.'
      : 'Transferring blendshapes... This may take a moment.',
    'info'
  );

  try {
    const form = new FormData();
    form.append('target', targetFile);
    if (refFile) {
      form.append('reference', refFile);
    }
    form.append('max_distance', document.getElementById('maxDistance').value);
    form.append('falloff_distance', document.getElementById('falloffDistance').value);

    const resp = await fetch('/api/transfer', { method: 'POST', body: form });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text);
    }

    resultBlob = await resp.blob();

    const arrayBuffer = await resultBlob.arrayBuffer();
    loadGLB(arrayBuffer, 'result_with_blendshapes.glb');

    showStatus('Transfer complete. Morph targets injected.', 'info');
    download.style.display = 'block';
  } catch (err) {
    showStatus(`Transfer failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Transfer Blendshapes';
    progress.classList.remove('active');
    updateTransferBtn();
  }
}

function downloadResult() {
  if (!resultBlob) return;
  const url = URL.createObjectURL(resultBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'avatar_with_blendshapes.glb';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Helpers ──
function showStatus(msg, type) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = `status visible ${type}`;
}

function updateTransferBtn() {
  // Only target is required — reference is optional (built-in fallback)
  document.getElementById('transferBtn').disabled = !targetFile;
}

// ── Init ──
initScene();

// Reference upload (optional — overrides built-in)
setupUploadZone('refUpload', async (file) => {
  refFile = file;
  document.getElementById('refFilename').textContent = file.name;
  document.getElementById('refUpload').classList.add('loaded');
  document.querySelector('#refUpload .label').textContent = 'Custom reference loaded';

  try {
    const info = await inspectFile(file);
    renderMeshInfo(info, 'refInfo');
  } catch (e) {
    showStatus(`Inspect failed: ${e.message}`, 'error');
  }

  const buf = await file.arrayBuffer();
  loadGLB(buf, file.name);
  updateTransferBtn();
});

// Target upload (required)
setupUploadZone('targetUpload', async (file) => {
  targetFile = file;
  document.getElementById('targetFilename').textContent = file.name;
  document.getElementById('targetUpload').classList.add('loaded');

  try {
    const info = await inspectFile(file);
    renderMeshInfo(info, 'targetInfo');
  } catch (e) {
    showStatus(`Inspect failed: ${e.message}`, 'error');
  }

  const buf = await file.arrayBuffer();
  loadGLB(buf, file.name);
  updateTransferBtn();
});

document.getElementById('transferBtn').addEventListener('click', doTransfer);
document.getElementById('downloadBtn').addEventListener('click', downloadResult);

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeTab = tab.dataset.tab;
    buildSliders();
  });
});

const viewport = document.querySelector('.viewport');
viewport.addEventListener('dragover', (e) => e.preventDefault());
viewport.addEventListener('drop', async (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.glb')) {
    const buf = await file.arrayBuffer();
    loadGLB(buf, file.name);
  }
});
