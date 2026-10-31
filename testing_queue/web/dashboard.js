'use strict';
const $ = id => document.getElementById(id);
let scopes = [], sessionRevision = 0, listRevision = 0, submitting = false;
class StaleResponse extends Error {}
function message(text) { $('message').textContent = text; }
function checkCurrent(revision) { if (revision !== sessionRevision) throw new StaleResponse(); }
function report(error) { if (!(error instanceof StaleResponse)) message(error.message); }
function disconnect(clearToken = false) {
  sessionRevision++; listRevision++; scopes = []; submitting = false;
  if (clearToken) $('token').value = '';
  $('submit').disabled = true;
  $('identity').textContent = 'Disconnected. Connect with a scoped token.';
  $('jobs').replaceChildren(); $('detail').textContent = 'No job selected.'; message('');
}
async function api(path, options = {}, revision = sessionRevision) {
  checkCurrent(revision);
  const token = $('token').value;
  if (!token) throw Error('Enter a scoped Authorization token.');
  const response = await fetch(path, {...options, headers: {Authorization: `Bearer ${token}`, ...options.headers}, credentials: 'omit', cache: 'no-store', redirect: 'error'});
  const data = await response.json();
  checkCurrent(revision);
  if (!response.ok) throw Error(data.message || data.error || `HTTP ${response.status}`);
  return data;
}
function display(value) { return value === null || value === undefined ? 'UNKNOWN' : String(value); }
async function refresh(revision = sessionRevision) {
  const sequence = ++listRevision;
  const data = await api('/v1/jobs', {}, revision);
  if (sequence !== listRevision) throw new StaleResponse();
  $('jobs').replaceChildren();
  for (const job of data.jobs) {
    const b = job.body, m = job.result?.metrics || {}, tr = document.createElement('tr');
    const runtime = m.runtime_kind === 'right_censored' ? `≥ ${display(m.observed_runtime_s)} s (censored)` : m.command_runtime_s != null ? `${m.command_runtime_s} s` : m.observed_runtime_s != null ? `${m.observed_runtime_s} s (${m.runtime_kind})` : 'UNKNOWN';
    const values = [`${b.candidate_id}\n${job.id}`, `${b.backend}\n${b.baseline_id}`, job.state, runtime, m.peak_memory_mib == null ? 'UNKNOWN' : `${m.peak_memory_mib} (${m.peak_kind})`, display(m.eval_loss), m.own_kernel_share == null ? 'UNKNOWN' : `${(m.own_kernel_share * 100).toFixed(2)}%`, `${m.admission || 'UNKNOWN'} / ${m.score == null ? '—' : m.score}`];
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); }
    tr.addEventListener('click', () => { $('detail').textContent = JSON.stringify(job, null, 2); });
    $('jobs').appendChild(tr);
  }
  message(`${data.jobs.length} job(s). Results refreshed; no work was launched.`);
}
$('token').addEventListener('input', () => disconnect());
$('clear').addEventListener('click', () => disconnect(true));
$('connect').addEventListener('click', async () => {
  disconnect(); const revision = sessionRevision;
  try {
    const session = await api('/v1/session', {}, revision);
    scopes = session.scopes;
    $('identity').textContent = `${session.subject} · ${session.role}`;
    $('submit').disabled = !scopes.includes('submit');
    await refresh(revision);
  } catch (error) { if (revision === sessionRevision) { scopes = []; $('submit').disabled = true; report(error); } }
});
$('refresh').addEventListener('click', () => refresh().catch(report));
$('upload').addEventListener('submit', async event => {
  event.preventDefault();
  if (submitting) return;
  if (!scopes.includes('submit')) return message('Submitter scope required.');
  const revision = sessionRevision; submitting = true; $('submit').disabled = true;
  try {
    const file = $('artifact').files[0];
    if (!file || file.size > 8 * 1024 * 1024) throw Error('Choose a source artifact up to 8 MiB.');
    const kind = file.name.endsWith('.ipynb') ? 'notebook' : file.name.endsWith('.zip') ? 'solution_bundle' : null;
    if (!kind) throw Error('Only .ipynb or .zip supported.');
    const bytes = new Uint8Array(await file.arrayBuffer()); checkCurrent(revision);
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
    const body = {candidate_id: $('candidate').value, backend: $('backend').value, baseline_id: $('baseline').value, entrypoint: $('entrypoint').value, artifact_kind: kind, priority: Number($('priority').value), contains_secrets: false, artifact_base64: btoa(binary)};
    const receipt = await api('/v1/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}, revision);
    await refresh(revision);
    message(`Submitted ${receipt.job_id}\nSource SHA256: ${receipt.source_sha256}\n${receipt.deduplicated ? 'Existing identical job returned.' : 'Awaiting independent review.'}`);
  } catch (error) { if (revision === sessionRevision) report(error); }
  finally { if (revision === sessionRevision) { submitting = false; $('submit').disabled = !scopes.includes('submit'); } }
});
