const $ = (id) => document.getElementById(id);
const KEYS = { apiBase: "", aiBand: 0.9 };

function health() {
  const el = $("status");
  el.className = ""; el.textContent = "checking…";
  chrome.runtime.sendMessage({ type: "health" }, (d) => {
    if (chrome.runtime.lastError || !d) { el.className = "bad"; el.textContent = "no response"; return; }
    if (!d.ok) { el.className = "bad"; el.textContent = d.error || "unreachable"; return; }
    el.className = "ok";
    el.textContent = "● " + (d.parameters / 1e6).toFixed(1) + "M params · " +
                     d.device + " · threshold " + d.threshold;
  });
}

function save(patch) { chrome.storage.sync.set(patch); }

chrome.storage.sync.get(KEYS, (s) => {
  const v = { ...KEYS, ...s };
  $("api").value = v.apiBase;
  $("band").value = v.aiBand;
  health();
});

$("api").addEventListener("change", (e) => { save({ apiBase: e.target.value.trim() }); health(); });
$("band").addEventListener("change", (e) => save({ aiBand: Math.min(0.99, Math.max(0.5, +e.target.value || 0.9)) }));
