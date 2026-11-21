'use strict';
const graph=window.SOUP_BASELINE;
if(graph.configurations.length!==1||graph.configurations[0].kind!=='baseline'||graph.edges.length||graph.runs.length||graph.comparisons.length)throw Error('The current graph must be baseline-only. Tested-node integration is not enabled.');
const node=graph.configurations[0];
if(graph.reference_anchor.z!==null||graph.reference_anchor.measured!==false)throw Error('Baseline admission and measurements must remain unknown.');
document.getElementById('ideasLink').href=graph.catalog.sheet_url;
document.getElementById('sourceLink').href='https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/'+graph.protocol.harness_commit+'/harness/run_task.py';
document.getElementById('hash').textContent='Configuration SHA-256: '+node.content_sha256;
const toggle=document.getElementById('detailsToggle'),details=document.getElementById('details');
toggle.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')==='true';toggle.setAttribute('aria-expanded',String(!open));details.hidden=open;toggle.textContent=open?'Show fixed task & evidence':'Hide fixed task & evidence';});
const point=document.getElementById('baselinePoint');
function inspect(){document.getElementById('bucket').focus();}
point.addEventListener('click',inspect);point.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspect();}});
