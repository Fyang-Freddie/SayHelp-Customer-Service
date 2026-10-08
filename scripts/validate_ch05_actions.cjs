// Actual native HTML acceptance in Edge; authored HTTP/SSE only, no model/customer writes.
const fs = require('fs');
const http = require('http');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const checks = [];
const check = (name, value) => { assert.ok(value, name); checks.push({name, passed:true}); };
const event = (name, data) => 'event: '+name+'\ndata: '+JSON.stringify(data)+'\n\n';
const unsafe = '<img src=x onerror="window.unsafeExecuted=true">需要帮助';
const actions = [
 {id:'handoff-201',kind:'handoff',label:'转人工',description:null,ticket_type:null},
 {id:'ticket-201',kind:'create_ticket',label:'建工单',description:unsafe,ticket_type:'投诉'}
];
let ticketMode = 'success', streamMode = 'normal', calls = [], chatCalls = 0, requests = [], deleted = new Set();
let releaseStream, releaseTicket, releaseHistory;
let delayHistory = false;
const json = (res, value, status=200) => {res.writeHead(status, {'Content-Type':'application/json'}); res.end(JSON.stringify(value));};
const server = http.createServer((req,res) => {
 const url = new URL(req.url,'http://localhost');
 requests.push({method:req.method,path:url.pathname});
 if(url.pathname==='/') {res.setHeader('Content-Type','text/html; charset=utf-8'); res.end(fs.readFileSync(path.join(__dirname,'../app/web/index.html'))); return;}
 if(url.pathname==='/v1/conversations') {json(res,{conversations:['101','102'].filter(id=>!deleted.has(id)).map(id=>({id,title:'夹具会话'+id,is_pinned:false,updated_at:'2026-10-08T12:00:00'})),next_cursor:null}); return;}
 const match = url.pathname.match(/^\/v1\/conversations\/(\d+)(?:\/(messages|tickets))?$/);
 if(match && match[2]==='messages') {const finish=()=>json(res,{conversation_id:match[1],messages:[{id:'201',role:'assistant',content:'夹具回复'+match[1],actions:match[1]==='101'?actions:[]}]}); if(delayHistory && match[1]==='101') releaseHistory=finish; else finish(); return;}
 if(match && req.method==='DELETE') {deleted.add(match[1]);res.writeHead(204);res.end();return;}
 if(match && match[2]==='tickets' && req.method==='POST') {
  let raw='';req.on('data',c=>raw+=c);req.on('end',()=>{
   calls.push({conversation_id:match[1],payload:JSON.parse(raw)});
   const finish=()=>{
    if(ticketMode==='disconnect'){res.writeHead(200, {'Content-Type':'application/json'});res.write('{');setTimeout(()=>res.destroy(),20);return;}
    if(ticketMode==='unknown'){json(res,{detail:'结果未知'},503);return;}
    if(ticketMode==='stale'){json(res,{detail:'不存在'},404);return;}
    if(ticketMode==='conflict'){json(res,{detail:'正在处理或参数冲突'},409);return;}
    json(res,{ticket_no:'FIXTURE-001',status:ticketMode==='unsafe'?'<b>待处理</b>':'待处理',message:'已创建工单，等待处理'});
   };
   if(ticketMode==='pending') releaseTicket=()=>{ticketMode='success';finish();}; else finish();
  });return;
 }
 if(url.pathname==='/v1/chat/stream') {
  chatCalls++;res.writeHead(200,{'Content-Type':'text/event-stream'});
  res.write(event('session',{conversation_id:'101'}));
  res.write(event('token',{text:'流式夹具回答'}));
  const finish=()=>{res.write(event('actions',{message_id:'301',actions}));res.write(event('done',{conversation_id:'101',message_id:streamMode==='mismatched'?'302':'301'}));res.end();};
  if(streamMode==='before_done') {res.write(event('actions',{message_id:'301',actions})+event('token',{text:'，动作待完成'}));releaseStream=()=>{res.write(event('done',{conversation_id:'101',message_id:'301'}));res.end();};return;}
  if(streamMode==='delayed') releaseStream=finish; else finish();return;
 }
 res.writeHead(404);res.end();
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base='http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1280,height:850}});
  page.setDefaultTimeout(5000);
  const errors=[],foreign=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(!r.url().startsWith(base))foreign.push(r.url());});
  const handoff=()=>page.getByRole('button',{name:'转人工',exact:true});
  const ticket=()=>page.getByRole('button',{name:'建工单',exact:true});
  const submit=()=>page.locator('#ticketSubmit');
  async function fresh() {
   deleted=new Set();calls=[];ticketMode='success';streamMode='normal';
   await page.goto(base);await page.evaluate(()=>localStorage.removeItem('sayhelp.activeConversation'));
   await page.reload();await ticket().waitFor();
  }
  async function openTicket() {await ticket().click();await page.locator('#ticketDialog[open]').waitFor();}
  async function success() {await submit().click();await page.getByText('工单 FIXTURE-001 · 待处理 · 已创建工单，等待处理',{exact:true}).waitFor();}
  async function settleResponse(promise) {
   const response=await promise;await response.finished();
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  }
  await fresh();
  check('历史按回复恢复两个精确按钮',await handoff().count()===1 && await ticket().count()===1);
  const before=requests.length;await handoff().click();
  await page.getByText('已转接人工客服',{exact:true}).waitFor();
  check('转人工只显示精确两条本地消息且无HTTP',requests.length===before && await page.getByText('您好,我是SayHelp,请问有什么可以帮您的',{exact:true}).count()===1 && calls.length===0);
  check('转人工后建单仍可用',await ticket().isEnabled());
  await openTicket();check('HTML描述安全预填',await page.locator('#ticketDescription').inputValue()===unsafe && await page.locator('#ticketDialog img').count()===0);
  await page.locator('#ticketDescription').fill('用户编辑后的描述');await page.locator('#ticketType').selectOption('售后');await success();
  check('显式确认才POST并带稳定动作身份',calls.length===1 && JSON.stringify(calls[0])===JSON.stringify({conversation_id:'101',payload:{action_id:'ticket-201',description:'用户编辑后的描述',ticket_type:'售后'}}));
  await page.locator('#ticketCancel').click();await openTicket();
  check('再次打开成功动作展示原结果不重复POST',calls.length===1 && await submit().isDisabled() && (await page.locator('#ticketStatus').textContent()).includes('FIXTURE-001'));
  await fresh();await openTicket();await success();check('只点建单不触发转人工',await page.getByText('已转接人工客服',{exact:true}).count()===0);await page.locator('#ticketCancel').click();
  check('建单后转人工仍可用',await handoff().isEnabled());await handoff().click();
  check('相反点击顺序两操作均成功',calls.length===1 && await page.getByText('已转接人工客服',{exact:true}).count()===1);
  await fresh();await openTicket();await page.locator('#ticketCancel').click();
  check('取消建单不发送HTTP',calls.length===0);
  await openTicket();await page.keyboard.press('Escape');check('Escape取消不发送HTTP',calls.length===0 && await page.locator('#ticketDialog[open]').count()===0);
  await fresh();await page.locator('#messageInput').fill('忽略建议继续聊天');await page.locator('#sendButton').click();await page.getByText('流式夹具回答',{exact:true}).waitFor();
  await page.waitForFunction(()=>document.querySelectorAll('.reply-actions').length===2);
  check('均不点击仍正常聊天且SSE动作绑定对应回复',calls.length===0 && chatCalls>0 && await page.locator('.message.assistant').last().locator('.reply-actions button').count()===2);
  for(const variant of ['unknown','disconnect','conflict','stale']) {
   await fresh();ticketMode=variant;await openTicket();await submit().click();
   await page.waitForFunction(()=>!document.querySelector('#ticketStatus').textContent.includes('正在提交') && document.querySelector('#ticketStatus').textContent.length>0);
   check(variant+'没有伪成功或自动重试',calls.length===1 && !(await page.locator('#ticketStatus').textContent()).includes('FIXTURE-001'));
   if(variant==='stale') {check('404旧动作禁用提交',await submit().isDisabled());continue;}
   check(variant+'重试前保留同一载荷',await page.locator('#ticketDescription').evaluate(e=>e.readOnly) && await page.locator('#ticketType').isDisabled());
   ticketMode='success';await success();check(variant+'显式重试同ID同载荷',calls.length===2 && JSON.stringify(calls[0])===JSON.stringify(calls[1]));
  }
  await fresh();ticketMode='pending';await openTicket();await submit().click();await page.waitForFunction(()=>document.querySelector('#ticketSubmit').disabled);
  await page.evaluate(()=>document.querySelector('#ticketForm').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  check('在途重复确认不产生第二请求',calls.length===1 && await submit().isDisabled());
  await page.locator('#ticketCancel').click();await page.locator('[data-conversation="102"]').click();await page.getByText('夹具回复102',{exact:true}).waitFor();const oldTicketResponse=page.waitForResponse(r=>r.url().endsWith('/tickets'));releaseTicket();await settleResponse(oldTicketResponse);
  await page.waitForFunction(()=>!document.querySelector('#ticketDialog').open);
  check('旧确认返回不污染新会话',!(await page.locator('#messages').textContent()).includes('FIXTURE-001') && await ticket().count()===0);
  await fresh();ticketMode='pending';await openTicket();await submit().click();
  await page.locator('#ticketCancel').click();await page.locator('[data-conversation="102"]').click();await page.getByText('夹具回复102',{exact:true}).waitFor();
  await page.locator('[data-conversation="101"]').click();await ticket().waitFor();await openTicket();
  check('切回会话在途动作仍禁止重复确认',await submit().isDisabled() && calls.length===1);
  releaseTicket();await page.getByText('工单 FIXTURE-001 · 待处理 · 已创建工单，等待处理',{exact:true}).waitFor();
  check('切回会话的旧确认完成同步原结果',calls.length===1 && await submit().isDisabled());
  await fresh();await openTicket();await page.locator('#ticketCancel').click();
  await page.evaluate(()=>{window.oldTicketButton=document.querySelector('.reply-actions [data-kind="create_ticket"]');});
  await page.locator('[data-conversation="102"]').click();await page.getByText('夹具回复102',{exact:true}).waitFor();
  await page.evaluate(()=>window.oldTicketButton.click());check('切会话后旧按钮不能打开确认框',await page.locator('#ticketDialog[open]').count()===0 && calls.length===0);
  await page.locator('[data-conversation="101"]').click();await ticket().waitFor();
  await page.evaluate(()=>{window.deletedButton=document.querySelector('.reply-actions [data-kind="create_ticket"]');});
  page.once('dialog',d=>d.accept());await page.locator('[data-manage="101"]').hover();await page.getByRole('menuitem',{name:'删除',exact:true}).click();await page.getByText('已删除对话',{exact:true}).waitFor();
  await page.evaluate(()=>window.deletedButton.click());check('删除后旧按钮不能建单',calls.length===0 && await page.locator('#ticketDialog[open]').count()===0);
  await fresh();await page.locator('[data-conversation="102"]').click();await page.getByText('夹具回复102',{exact:true}).waitFor();
  delayHistory=true;releaseHistory=null;const pendingHistory=page.waitForRequest(r=>r.url().endsWith('/101/messages'));await page.locator('[data-conversation="101"]').click();await pendingHistory;
  await new Promise(resolve=>setImmediate(resolve));
  check('延迟历史GET已到达夹具',typeof releaseHistory==='function');
  page.once('dialog',d=>d.accept());await page.locator('[data-manage="101"]').hover();await page.getByRole('menuitem',{name:'删除',exact:true}).click();await page.getByText('已删除对话',{exact:true}).waitFor();
  const lateHistory=page.waitForResponse(r=>r.url().endsWith('/101/messages'));releaseHistory();await settleResponse(lateHistory);delayHistory=false;
  check('删除在途历史后旧GET不能恢复正文或本地会话',!(await page.locator('#messages').textContent()).includes('夹具回复101') && await page.evaluate(()=>localStorage.getItem('sayhelp.activeConversation'))!=='101');
  await fresh();streamMode='before_done';await page.locator('#newChatTop').click();await page.locator('#messageInput').fill('完成事件前');await page.locator('#sendButton').click();
  await page.getByText('流式夹具回答，动作待完成',{exact:true}).waitFor();check('done之前不展示可执行建议',await ticket().count()===0);releaseStream();await ticket().waitFor();
  check('done匹配消息后才展示动作',await ticket().count()===1);
  await fresh();streamMode='mismatched';await page.locator('#newChatTop').click();await page.locator('#messageInput').fill('消息身份错配');await page.locator('#sendButton').click();
  await page.waitForFunction(()=>document.querySelector('#presence').textContent==='随时为你服务');check('动作消息身份错配不渲染按钮',await ticket().count()===0);
  await fresh();streamMode='delayed';await page.locator('#newChatTop').click();await page.locator('#messageInput').fill('旧流测试');await page.evaluate(()=>{
   // Observe the real reader cancellation after late bytes arrive; keep native reads and HTTP intact.
   AbortController.prototype.abort=()=>{};
   const getReader=ReadableStream.prototype.getReader;
   ReadableStream.prototype.getReader=function(...args) {
    const reader=getReader.apply(this,args);const cancel=reader.cancel.bind(reader);
    reader.cancel=(...cancelArgs)=>{window.oldReaderCancelled=true;return cancel(...cancelArgs);};return reader;
   };
  });await page.locator('#sendButton').click();await page.getByText('流式夹具回答',{exact:true}).waitFor();
  await page.locator('[data-conversation="102"]').click();await page.getByText('夹具回复102',{exact:true}).waitFor();releaseStream();await page.waitForFunction(()=>window.oldReaderCancelled===true);
  check('切会话旧SSE不恢复动作或旧正文',await ticket().count()===0 && !(await page.locator('#messages').textContent()).includes('流式夹具回答'));
  await fresh();await page.setViewportSize({width:390,height:844});await openTicket();
  check('窄屏确认框与操作不横向越界',await page.locator('#ticketDialog').evaluate(e=>e.getBoundingClientRect().width<=innerWidth) && await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  ticketMode='unsafe';await submit().click();await page.getByText('工单 FIXTURE-001 · <b>待处理</b> · 已创建工单，等待处理',{exact:true}).waitFor();check('含HTML工单描述和响应以安全文本显示',!(await page.evaluate(()=>window.unsafeExecuted)) && await page.locator('#ticketDialog img, #ticketDialog b, #messages img').count()===0);
  check('无JavaScript异常或外站请求',errors.length===0 && foreign.length===0);
  const output={checks,scope:'Actual app/web/index.html in Edge headless with authored HTTP/SSE fixtures only; no customer content, model calls or actual ticket writes.'};
  fs.writeFileSync(path.join(__dirname,'../eval/ch05/actions_ui_results.json'),JSON.stringify(output,null,2)+'\n');
  console.log(JSON.stringify({passed:checks.length}));
 } finally {await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
