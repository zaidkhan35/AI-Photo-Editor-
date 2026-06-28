// AI Photo Editor — Phase 6 Frontend

const API = "http://localhost:8000";

let uploadedFile = null;
let originalSrc  = null;
let resultSrc    = null;
let currentView  = "split";
let applyTimer   = null;

// ── ELEMENTS ──
const uploadZone        = document.getElementById("upload-zone");
const fileInput         = document.getElementById("file-input");
const btnApply          = document.getElementById("btn-apply");
const btnReset          = document.getElementById("btn-reset");
const btnDownload       = document.getElementById("btn-download");
const statusBadge       = document.getElementById("status-badge");
const spinner           = document.getElementById("spinner");
const emptyState        = document.getElementById("empty-state");
const splitView         = document.getElementById("split-view");
const viewOriginal      = document.getElementById("view-original");
const viewResult        = document.getElementById("view-result");
const imgOriginal       = document.getElementById("img-original");
const imgResult         = document.getElementById("img-result");
const imgOriginalSingle = document.getElementById("img-original-single");
const imgResultSingle   = document.getElementById("img-result-single");
const faceCount         = document.getElementById("face-count");
const previewInfo       = document.getElementById("preview-info");
const colorPreview      = document.getElementById("color-preview");

// ── UPLOAD ──
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    originalSrc = e.target.result;
    imgOriginal.src       = originalSrc;
    imgOriginalSingle.src = originalSrc;
    emptyState.classList.add("hidden");
    splitView.classList.remove("hidden");
    showView("split");
    btnApply.disabled = false;
    setStatus("ready", file.name);
    previewInfo.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

// ── SLIDERS — live display ──
document.querySelectorAll("input[type='range']").forEach(slider => {
  const valEl = document.getElementById(`val-${slider.id}`);
  if (valEl) valEl.textContent = slider.value;
  slider.addEventListener("input", () => {
    if (valEl) valEl.textContent = slider.value;
    updateColorPreview();
    scheduleApply();
  });
});

document.querySelectorAll("input[type='checkbox'], select").forEach(el => {
  el.addEventListener("change", scheduleApply);
});
document.getElementById("text_content").addEventListener("input", scheduleApply);

function scheduleApply() {
  if (!uploadedFile) return;
  clearTimeout(applyTimer);
  applyTimer = setTimeout(applyEdits, 600);
}

// ── COLOR PREVIEW ──
function updateColorPreview() {
  const r = document.getElementById("text_r").value;
  const g = document.getElementById("text_g").value;
  const b = document.getElementById("text_b").value;
  colorPreview.style.background = `rgb(${r},${g},${b})`;
}
updateColorPreview();

// ── PHASE TABS ──
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".phase-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`phase-${tab.dataset.phase}`).classList.add("active");
  });
});

// ── PREVIEW TABS ──
document.querySelectorAll(".ptab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".ptab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    showView(tab.dataset.view);
  });
});

function showView(view) {
  currentView = view;
  splitView.classList.add("hidden");
  viewOriginal.classList.add("hidden");
  viewResult.classList.add("hidden");
  if (view === "split"    && originalSrc) splitView.classList.remove("hidden");
  if (view === "original" && originalSrc) viewOriginal.classList.remove("hidden");
  if (view === "result"   && resultSrc)   viewResult.classList.remove("hidden");
}

// ── GET STATE ──
function getState() {
  const v = id => parseInt(document.getElementById(id)?.value || 0);
  const c = id => document.getElementById(id)?.checked ? 1 : 0;
  return {
    brightness: v("brightness"),  contrast: v("contrast"),
    blur:       v("blur"),        sharpen:  c("sharpen"),
    grayscale:  c("grayscale"),   scale:    v("scale"),
    rotate:     v("rotate"),      flip:     v("flip"),
    crop_x:     v("crop_x"),      crop_y:   v("crop_y"),
    crop_w:     v("crop_w"),      crop_h:   v("crop_h"),

    cartoon:      v("cartoon"),      sketch_bw:    v("sketch_bw"),
    sketch_color: v("sketch_color"), stylize:      v("stylize"),
    edges:        v("edges"),        emboss:       v("emboss"),
    sepia:        v("sepia"),

    face_mode:        v("face_mode"),
    min_neighbors:    v("min_neighbors"),
    scale_factor_x10: v("scale_factor_x10"),

    bg_remove:    c("bg_remove"),  denoise:      v("denoise"),
    auto_enhance: v("auto_enhance"), super_res:  v("super_res"),

    text_on:      c("text_on"),
    text_content: document.getElementById("text_content")?.value || "",
    text_x:       v("text_x"),    text_y:    v("text_y"),
    text_size:    v("text_size"), text_r:    v("text_r"),
    text_g:       v("text_g"),    text_b:    v("text_b"),
    enhance_qual: v("enhance_qual"),
  };
}

// ── APPLY ──
btnApply.addEventListener("click", applyEdits);

async function applyEdits() {
  if (!uploadedFile) return;

  setStatus("loading", "Processing…");
  spinner.classList.remove("hidden");

  const state    = getState();
  const formData = new FormData();
  formData.append("image", uploadedFile);
  for (const [key, val] of Object.entries(state)) {
    formData.append(key, val);
  }

  try {
    const res  = await fetch(`${API}/process`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) { setStatus("error", data.error); return; }

    resultSrc = data.result;
    imgResult.src       = resultSrc;
    imgResultSingle.src = resultSrc;

    faceCount.textContent    = state.face_mode > 0 ? data.face_count : "—";
    previewInfo.textContent  = `${data.width} × ${data.height}px`;
    btnDownload.disabled     = false;
    setStatus("ready", "Done");

    if (currentView === "result") showView("result");

  } catch (err) {
    setStatus("error", "Cannot reach backend — is it running?");
  } finally {
    spinner.classList.add("hidden");
  }
}

// ── DOWNLOAD ──
btnDownload.addEventListener("click", () => {
  if (!resultSrc) return;
  const a = document.createElement("a");
  a.href = resultSrc; a.download = "edited_photo.jpg"; a.click();
});

// ── RESET ──
btnReset.addEventListener("click", () => {
  const s = (id, v) => { const el = document.getElementById(id); const d = document.getElementById(`val-${id}`); if(el) el.value=v; if(d) d.textContent=v; };
  const c = (id, v) => { const el = document.getElementById(id); if(el) el.checked=v; };

  s("brightness",100); s("contrast",10); s("blur",1); s("scale",100);
  s("rotate",0); s("crop_x",0); s("crop_y",0); s("crop_w",0); s("crop_h",0);
  c("sharpen",false); c("grayscale",false);
  document.getElementById("flip").value = "0";

  ["cartoon","sketch_bw","sketch_color","stylize","edges","emboss","sepia"]
    .forEach(id => s(id, 0));

  document.getElementById("face_mode").value = "0";
  s("min_neighbors",5); s("scale_factor_x10",12);

  c("bg_remove",false);
  s("denoise",0); s("auto_enhance",0); s("super_res",0);

  c("text_on",false);
  document.getElementById("text_content").value = "Hello World";
  s("text_x",10); s("text_y",10); s("text_size",40);
  s("text_r",255); s("text_g",255); s("text_b",255);
  s("enhance_qual",0);

  updateColorPreview();
  faceCount.textContent = "—";
});

// ── STATUS ──
function setStatus(type, msg) {
  statusBadge.className   = `badge badge-${type}`;
  statusBadge.textContent = msg;
}