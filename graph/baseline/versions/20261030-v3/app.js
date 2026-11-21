'use strict';
const graph=window.SOUP_BASELINE;
if(graph.configurations.length!==1||graph.configurations[0].kind!=='baseline'||graph.edges.length||graph.comparisons.length||graph.runs.length!==1||graph.runs[0].role!=='base')throw Error('The current graph must contain only B0 and its one baseline observation. Candidate integration is not enabled.');
const node=graph.configurations[0],run=graph.runs[0];
if(graph.reference_anchor.z!==null||run.admission_status!=='unknown'||run.official_points!==null)throw Error('Baseline admission and official points must remain unknown.');
if(run.status!=='completed'||run.repeats!==1||run.public_eval!==true||run.config_id!==node.id)throw Error('Expected the verified single-repeat public baseline observation.');
document.getElementById('ideasLink').href=graph.catalog.sheet_url;
document.getElementById('sourceLink').removeAttribute('href');
document.getElementById('sourceLink').setAttribute('aria-disabled', 'true');
document.getElementById('hash').textContent='Configuration SHA-256: '+node.content_sha256+' · Evidence verified '+run.verified_at;
document.getElementById('wallTime').textContent=run.wall_s.toFixed(2)+' s';
document.getElementById('peakMemory').textContent=run.gpu_peak_mib.toLocaleString('en-US')+' MiB';
document.getElementById('evalLoss').textContent=run.eval_loss.toFixed(5);
document.getElementById('loopSteps').textContent=run.train_loop_s.toFixed(2)+' s / '+run.steps+' steps';
const toggle=document.getElementById('detailsToggle'),details=document.getElementById('details');
toggle.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')==='true';toggle.setAttribute('aria-expanded',String(!open));details.hidden=open;toggle.textContent=open?'Show fixed task & evidence':'Hide fixed task & evidence';});
const point=document.getElementById('baselinePoint');
function inspect(){document.getElementById('bucket').focus();}
point.addEventListener('click',inspect);point.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspect();}});
