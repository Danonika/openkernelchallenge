'use strict';
let token = '', timer = null, busy = false, generation = 0, canSubmit = false, jobs = [];
const expanded = new Set();
const reruns = new Map();
const $ = id => document.getElementById(id);
const show = value => { $('status').textContent = value; };
const el = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
const finite = n => typeof n === 'number' && Number.isFinite(n);
const fmt = (n, digits = 2) => finite(n) ? n.toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits}) : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
const percent = n => finite(n) ? (n > 0 ? '+' : '') + fmt(n, 2) + '%' : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
const failed = job => ['failed', 'timed_out', 'interrupted', 'cancelled'].includes(job.state);
const active = job => ['queued', 'running'].includes(job.state);
const outcomeText = {
  beats_baseline: 'Beat baseline', faster_more_memory: 'Faster Р В РІР‚в„ўР вЂ™Р’В· more VRAM',
  less_memory: 'Less VRAM Р В РІР‚в„ўР вЂ™Р’В· no speed gain', no_gain: 'No measured gain',
  checks_failed: 'Checks failed', incomplete: 'Incomplete comparison',
  no_metrics: 'No training report', run_failed: 'Run failed', pending: 'In progress'
};
async function request(path, body) {
  if (!token) throw new Error('Connect with an access token first.');
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST',
    headers: {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body), credentials: 'omit', redirect: 'error'});
  if (!response.ok) {
    const detail = await response.json();
    throw new Error(detail.message || detail.error || ('HTTP ' + response.status));
  }
  return response;
}
function metricCell(value, suffix, note, digits = 2) {
  const cell = el('td', undefined, 'numeric');
  cell.append(el('span', finite(value) ? fmt(value, digits) + suffix : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ', 'primary'));
  if (note) cell.append(el('span', note, 'sub'));
  return cell;
}
async function download(job) {
  const current = generation;
  try {
    const blob = await (await request(job.outputs_url)).blob();
    if (current !== generation) return;
    const url = URL.createObjectURL(blob), link = el('a');
    link.href = url; link.download = job.id + '-outputs.zip'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { if (current === generation) show(error.message); }
}
function detailsRow(job) {
  const row = el('tr', undefined, 'detail-row'), cell = el('td');
  cell.colSpan = 8;
  const grid = el('div', undefined, 'detail-grid'), left = el('div'), right = el('div');
  const m = job.metrics || {}, b = m.base || {}, s = m.solution || {};
  left.append(el('strong', job.id), el('p', 'Submitted ' + new Date(job.created * 1000).toLocaleString()));
  if (job.result) left.append(el('p', 'Whole job: ' + fmt(job.result.elapsed_s) + ' s Р В РІР‚в„ўР вЂ™Р’В· includes warm-up, training and evaluation'));
  if (job.rerun_of) left.append(el('p', 'Re-run of ' + job.rerun_of));
  if (job.result && job.result.error) left.append(el('p', job.result.error.code + ': ' + job.result.error.message, 'detail-error'));
  else if (job.result) left.append(el('p', job.result.reason));
  right.append(el('strong', 'Matching baseline comparison'));
  if (m.baseline_source === 'saved') right.append(el('p', 'Saved baseline: ' + m.baseline_job_id + ' Р В РІР‚в„ўР вЂ™Р’В· no extra base run'));
  right.append(el('p', 'Training: base ' + fmt(b.wall_s) + ' s Р В Р вЂ Р Р†Р вЂљР’В Р Р†Р вЂљРІвЂћСћ solution ' + fmt(s.wall_s) + ' s'));
  right.append(el('p', 'GPU peak: base ' + fmt(b.gpu_peak_mib, 0) + ' MiB Р В Р вЂ Р Р†Р вЂљР’В Р Р†Р вЂљРІвЂћСћ solution ' + fmt(s.gpu_peak_mib, 0) + ' MiB'));
  right.append(el('p', 'Eval loss: base ' + fmt(b.eval_loss, 5) + ' Р В Р вЂ Р Р†Р вЂљР’В Р Р†Р вЂљРІвЂћСћ solution ' + fmt(s.eval_loss, 5) + ' Р В РІР‚в„ўР вЂ™Р’В· maximum ' + fmt(m.loss_limit, 5)));
  if (s.partial) right.append(el('p', 'Training stopped early. Time and GPU peak are measurements up to the stop; speedup is an upper bound.'));
  if (!finite(s.eval_loss)) right.append(el('p', 'Solution not evaluated: no completed evaluation loss was recorded.'));
  if (finite(m.host_gpu_peak_mib)) right.append(el('p', 'Whole-notebook GPU peak: ' + fmt(m.host_gpu_peak_mib, 0) + ' MiB (includes warm-up and evaluation)'));
  right.append(el('p', 'Beat baseline = faster training, no more GPU memory, public loss/resource checks passed, environment unchanged.'));
  grid.append(left, right); cell.append(grid);
  const raw = el('details'); raw.append(el('summary', 'API metrics and failure details'), el('pre', JSON.stringify({metrics: m, result: job.result}, null, 2)));
  cell.append(raw); row.append(cell); return row;
}
function jobRow(job) {
  const row = el('tr', undefined, 'job-row'); row.dataset.jobId = job.id;
  const m = job.metrics || {}, b = m.base || {}, s = m.solution || {}, code = m.comparison || 'no_metrics';
  const d = m.display || s, partial = m.speedup_kind === 'upper_bound';
  const identity = el('td');
  identity.append(el('span', job.candidate, 'candidate'), el('div', job.id.slice(0, 14) + 'Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В¦', 'job-id'));
  identity.append(el('span', job.state.replaceAll('_', ' '), 'pill ' + (failed(job) ? 'bad' : active(job) ? 'running' : 'neutral')));
  row.append(identity);
  const outcome = el('td');
  outcome.append(el('div', outcomeText[code] || 'No comparison', 'outcome ' + (code === 'beats_baseline' ? 'good-text' : code === 'checks_failed' || failed(job) ? 'bad-text' : '')));
  if (job.result && job.result.error) outcome.append(el('span', job.result.error.code, 'sub failure-text'));
  else if (code === 'incomplete') outcome.append(el('span', 'Missing compatible evidence', 'sub'));
  row.append(outcome);
  row.append(metricCell(d.wall_s, ' s', (d.partial ? 'Stopped Р В РІР‚в„ўР вЂ™Р’В· ' : '') + (d.side === 'base' ? 'base run' : finite(b.wall_s) ? 'base ' + fmt(b.wall_s) + ' s' : 'No matching baseline')));
  const speed = metricCell(m.speedup, 'Р В РІР‚СљР Р†Р вЂљРІР‚Сњ', partial ? 'Upper bound Р В РІР‚в„ўР вЂ™Р’В· stopped early' : finite(m.speedup) ? (m.baseline_source === 'saved' ? 'Saved baseline Р В РІР‚в„ўР вЂ™Р’В· ' : '') + (m.speedup > 1 ? fmt(100 * (1 - 1 / m.speedup), 1) + '% less time' : m.speedup < 1 ? fmt(100 * (1 / m.speedup - 1), 1) + '% more time' : 'Same time') : 'No matching training time');
  if (partial && finite(m.speedup)) speed.firstChild.textContent = 'Р В Р вЂ Р Р†Р вЂљР’В°Р вЂ™Р’В¤ ' + speed.firstChild.textContent;
  if (finite(m.speedup)) speed.firstChild.classList.add(m.speedup >= 1 ? 'good-text' : 'bad-text');
  row.append(speed);
  row.append(metricCell(d.gpu_peak_mib, ' MiB', d.gpu_scope === 'whole_notebook' ? 'Whole notebook Р В РІР‚в„ўР вЂ™Р’В· host sampled' : d.partial ? 'Peak before stop' : finite(m.memory_saved_pct) ? fmt(Math.abs(m.memory_saved_pct), 1) + '% ' + (m.memory_saved_pct >= 0 ? 'less VRAM' : 'more VRAM') + ' Р В РІР‚в„ўР вЂ™Р’В· base ' + fmt(b.gpu_peak_mib, 0) : (d.side === 'base' ? 'Base training peak' : 'Training peak'), 0));
  const loss = metricCell(d.eval_loss, '', finite(m.loss_delta_pct) ? percent(m.loss_delta_pct) + ' vs base ' + fmt(b.eval_loss, 5) : finite(d.eval_loss) ? (d.side === 'base' ? 'Base evaluation' : 'Measured evaluation') : active(job) ? 'Awaiting evaluation' : 'No completed evaluation', 5);
  if (!finite(d.eval_loss)) loss.firstChild.textContent = 'Not evaluated';
  row.append(loss);
  const checks = el('td'), list = el('div', undefined, 'checks');
  const values = m.checks || {};
  for (const [key, label] of [['loss', 'Loss'], ['time', 'Time'], ['memory', 'VRAM']]) {
    let state = values[key] || 'unknown';
    const failure = job.result && job.result.error;
    if (failure && (key === 'loss' && ['EVAL_LOSS_EXCEEDED', 'INVALID_EVAL_RESULT'].includes(failure.code) ||
        key === 'time' && failure.code === 'TIME_LIMIT_EXCEEDED' && failure.phase === 'training' ||
        key === 'memory' && failure.code === 'MEMORY_LIMIT_EXCEEDED' && failure.unit === 'MiB')) state = 'failed';
    list.append(el('span', label + (state === 'passed' ? ' Р В Р вЂ Р РЋРЎв„ўР Р†Р вЂљРЎС™' : state === 'failed' ? ' Р В Р вЂ Р РЋРЎв„ўР Р†Р вЂљРЎС›' : ' Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ'), 'pill check-item ' + (state === 'passed' ? 'good' : state === 'failed' ? 'bad' : 'neutral')));
  }
  if (values.environment === 'changed') list.append(el('span', 'Env changed', 'pill check-item warn'));
  checks.append(list); row.append(checks);
  const actions = el('td'), buttons = el('div', undefined, 'actions'), detail = el('button', expanded.has(job.id) ? 'Close' : 'Details', 'secondary');
  detail.setAttribute('aria-expanded', String(expanded.has(job.id)));
  detail.onclick = () => { if (expanded.has(job.id)) expanded.delete(job.id); else expanded.add(job.id); render(); };
  buttons.append(detail);
  if (job.output_sha) { const button = el('button', 'Outputs', 'secondary'); button.title = 'Download outputs'; button.onclick = () => download(job); buttons.append(button); }
  if (canSubmit && !active(job)) {
    const attempt = reruns.get(job.id), rerun = el('button', attempt && attempt.busy ? 'QueuingР В Р вЂ Р В РІР‚С™Р вЂ™Р’В¦' : 'Re-run', 'secondary');
    rerun.disabled = !!(attempt && attempt.busy);
    rerun.title = 'Submit the saved notebook again with the same limits';
    rerun.onclick = async () => {
      const current = generation;
      const pending = reruns.get(job.id) || {requestId: crypto.randomUUID()};
      if (pending.busy) return;
      pending.busy = true; reruns.set(job.id, pending); render();
      try {
        const created = await (await request('/v1/notebooks/' + job.id + '/rerun', {request_id: pending.requestId})).json();
        reruns.delete(job.id);
        if (current !== generation) return;
        $('search').value = ''; $('filter').value = 'all'; $('sort').value = 'newest';
        show('Re-run queued: ' + created.id); await refresh();
      } catch (error) { if (current === generation) show(error.message); }
      finally { pending.busy = false; if (current === generation) render(); }
    };
    buttons.append(rerun);
  }
  if (canSubmit && active(job)) {
    const cancel = el('button', job.cancel_requested ? 'Cancelling' : 'Cancel', 'secondary');
    cancel.disabled = job.cancel_requested;
    cancel.onclick = async () => {
      const current = generation;
      try { await request('/v1/notebooks/' + job.id + '/cancel', {}); await refresh(); }
      catch (error) { if (current === generation) show(error.message); }
    };
    buttons.append(cancel);
  }
  actions.append(buttons); row.append(actions); return row;
}
function render() {
  $('stat-total').textContent = token ? jobs.length : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
  $('stat-active').textContent = token ? jobs.filter(active).length : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
  $('stat-wins').textContent = token ? jobs.filter(j => j.metrics && j.metrics.comparison === 'beats_baseline').length : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
  const eligible = jobs.filter(j => j.state === 'completed' && j.metrics && ['beats_baseline', 'faster_more_memory', 'less_memory', 'no_gain'].includes(j.metrics.comparison));
  const speeds = eligible.map(j => j.metrics.speedup).filter(finite);
  $('stat-speedup').textContent = speeds.length ? fmt(Math.max(...speeds)) + 'Р В РІР‚СљР Р†Р вЂљРІР‚Сњ' : 'Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ';
  const query = $('search').value.trim().toLowerCase(), filter = $('filter').value;
  let visible = jobs.filter(j => (!query || (j.candidate + ' ' + j.id).toLowerCase().includes(query)) &&
    (filter === 'all' || filter === 'active' && active(j) || filter === 'failed' && failed(j) ||
     filter === 'comparable' && j.metrics && (j.metrics.comparable || j.metrics.paired) || filter === 'wins' && j.metrics && j.metrics.comparison === 'beats_baseline'));
  const sort = $('sort').value;
  const value = j => sort === 'speedup' ? j.metrics && j.metrics.speedup : j.metrics && j.metrics.solution && j.metrics.solution[sort === 'memory' ? 'gpu_peak_mib' : 'eval_loss'];
  visible.sort((a, b) => {
    if (sort === 'newest') return b.created - a.created;
    const av = value(a), bv = value(b);
    if (!finite(av)) return finite(bv) ? 1 : b.created - a.created;
    if (!finite(bv)) return -1;
    return (sort === 'speedup' ? bv - av : av - bv) || b.created - a.created;
  });
  const fragment = document.createDocumentFragment();
  for (const job of visible) { fragment.append(jobRow(job)); if (expanded.has(job.id)) fragment.append(detailsRow(job)); }
  $('jobs').replaceChildren(fragment);
  $('empty').hidden = visible.length > 0;
  $('empty').textContent = !token ? 'Connect to see your experiments.' : jobs.length ? 'No jobs match these filters.' : 'No jobs yet. Submit your first notebook below.';
  $('row-count').textContent = token ? visible.length + ' of ' + jobs.length + ' jobs shown' : 'No jobs loaded';
}
async function refresh() {
  if (busy || !token) return;
  const current = generation; busy = true;
  try {
    const data = await (await request('/v1/notebooks')).json();
    if (current !== generation) return;
    jobs = data.jobs; render(); $('updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (error) { if (current === generation) show(error.message); }
  finally { busy = false; }
}
$('connect').onclick = async () => {
  const current = ++generation; clearInterval(timer); canSubmit = false;
  $('submit').querySelector('button[type=submit]').disabled = true;
  token = $('token').value.trim(); $('token').value = ''; jobs = []; expanded.clear(); render();
  try {
    const identity = await (await request('/v1/session')).json();
    if (current !== generation) return;
    const runtime = await (await request('/v1/notebook-runtime')).json();
    if (current !== generation) return;
    canSubmit = identity.scopes.includes('submit');
    $('submit').querySelector('button[type=submit]').disabled = !canSubmit;
    $('runtime').textContent = 'Tesla T4 Р В РІР‚в„ўР вЂ™Р’В· 4 CPUs Р В РІР‚в„ўР вЂ™Р’В· 30 GiB container limit\nHost physical RAM: ' + runtime.host_memory_gib + ' GiB\nPython ' + runtime.python + ' Р В РІР‚в„ўР вЂ™Р’В· PyTorch ' + runtime.pytorch + ' Р В РІР‚в„ўР вЂ™Р’В· Triton ' + runtime.triton +
      '\n40 s per training run | GPU peak <= 2,261 MiB | eval loss <= original baseline +0.5%' +
      (runtime.workspace_pool_gib ? '\n/tmp uses job disk Р В РІР‚в„ўР вЂ™Р’В· ' + runtime.workspace_pool_gib + ' GiB shared workspace pool\nShared memory: up to ' + runtime.shared_memory_limit_gib + ' GiB within the 30 GiB RAM limit' : '');
    show('Connected'); await refresh();
    if (current === generation) timer = setInterval(refresh, 5000);
  } catch (error) { if (current === generation) { token = ''; render(); show(error.message); } }
};
$('disconnect').onclick = () => {
  generation++; canSubmit = false; token = ''; clearInterval(timer); jobs = []; expanded.clear();
  $('submit').querySelector('button[type=submit]').disabled = true; $('token').value = '';
  $('runtime').textContent = 'Connect to inspect the pinned runtime.'; $('updated').textContent = 'Refreshes every 5 seconds';
  render(); show('Disconnected');
};
$('refresh').onclick = refresh;
$('search').oninput = render; $('filter').onchange = render; $('sort').onchange = render;
$('submit').onsubmit = async event => {
  event.preventDefault(); const current = generation, button = event.target.querySelector('button[type=submit]'); button.disabled = true;
  try {
    const file = $('file').files[0];
    if (!file || file.size > 8 * 1024 * 1024) throw new Error('Choose a notebook of at most 8 MiB.');
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (current !== generation || !canSubmit) return;
    let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    const timeout = Number($('timeout').value);
    const job = await (await request('/v1/notebooks', {candidate_id: $('candidate').value,
      request_id: crypto.randomUUID(), notebook_base64: btoa(binary), contains_secrets: false,
      timeout_s: timeout, cell_timeout_s: Math.min(timeout, 1800)})).json();
    if (current !== generation) return;
    show('Submitted ' + job.id); await refresh();
  } catch (error) { if (current === generation) show(error.message); }
  finally { button.disabled = !canSubmit; }
};
render();

