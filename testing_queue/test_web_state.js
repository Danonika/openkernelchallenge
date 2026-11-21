'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.value = ''; this.textContent = ''; this.disabled = true; this.children = []; this.handlers = {}; }
  addEventListener(name, fn) { this.handlers[name] = fn; }
  replaceChildren(...children) { this.children = children; }
  appendChild(child) { this.children.push(child); }
}
const elements = {}, pending = [];
const context = {
  document: {getElementById: id => elements[id] ||= new Element(), createElement: () => new Element()},
  fetch: (url, options) => new Promise(resolve => pending.push({url, options, resolve})),
  Uint8Array, String, Number, Error,
};
vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname + '/web/dashboard.js', 'utf8'), context);
const tick = () => new Promise(resolve => setImmediate(resolve));
const reply = data => { const request = pending.shift(); assert.ok(request); request.resolve({ok: true, json: async () => data}); return request; };
const session = {scopes: ['read','submit'], subject: 'api-submitter', role: 'submitter'};
const job = {id: 'q_fake', body: {candidate_id: '<script>not html</script>', backend: 'CPU', baseline_id: 'synthetic'}, state: 'submitted'};
async function connect() {
  elements.token.value = 'TEST_ONLY_FAKE_DOM_xxxxxxxxxxxxxxxxxxxxxxxx';
  const promise = elements.connect.handlers.click(); reply(session); await tick();reply({jobs: [job]});await promise;
}
(async () => {
  await connect();assert.equal(elements.submit.disabled, false);assert.equal(elements.jobs.children.length,1);
  assert.equal(elements.jobs.children[0].children[0].textContent, '<script>not html</script>\nq_fake');

  const refresh = elements.refresh.handlers.click();elements.clear.handlers.click();reply({jobs: [job]});await refresh;await tick();
  assert.equal(elements.jobs.children.length,0);assert.equal(elements.submit.disabled,true);assert.equal(elements.token.value,'');

  elements.token.value = 'TEST_ONLY_LATE_CONNECT_xxxxxxxxxxxxxxxxxxx';
  const late = elements.connect.handlers.click();elements.clear.handlers.click();reply(session);await late;
  assert.equal(elements.submit.disabled,true);assert.match(elements.identity.textContent,/Disconnected/);assert.equal(pending.length,0);

  await connect();const edited = elements.refresh.handlers.click();elements.token.value='different';elements.token.handlers.input();reply({jobs:[job]});await edited;await tick();
  assert.equal(elements.jobs.children.length,0);assert.equal(elements.submit.disabled,true);
  console.log('PASS: 4 shipped-JavaScript state checks (fake DOM, not rendering)');
})().catch(error => { console.error(error);process.exitCode=1; });
