// Browser acceptance of the actual chat HTML with authored HTTP/SSE fixtures.
// Usage: node scripts/validate_ch04_citations.cjs (Playwright must be available).
const fs = require('fs');
const http = require('http');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const checks=[];
const check=(name,value)=>{assert.ok(value,name);checks.push(name)};
const text='# 文档\n第一段原文\n第二段原文\n末尾原文';
const evidence=[1,3,10].map(n=>({n,chunk_id:String(n),question:'证据'+n,answer:'原文快照'+n+'<script>安全文本</script>',section_path:'手册 / 章节'+n,source_file:'knowledge_db/product-faq.md',source_url:'/v1/knowledge/documents/product-faq.md',source_start_line:2,source_end_line:3,source_digest:'same'}));
let mode='current', documentCalls=[], foreignCalls=[], readyAt=0;
const body='按规则收取[1]，会员权益[3]，配送范围[10]，不存在[99]。'+ '已校验文字。'.repeat(120);
const event=(name,data)=>'event: '+name+'\ndata: '+JSON.stringify(data)+'\n\n';
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');
 if(url.pathname==='/'){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(fs.readFileSync(path.join(__dirname,'../app/web/index.html')));return;}
 if(url.pathname==='/v1/conversations'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({conversations:mode==='current'||mode==='interrupted'?[]:[{id:'101',title:'验收会话',is_pinned:false,updated_at:'2026-10-06T12:00:00'}],next_cursor:null}));return;}
 if(url.pathname.endsWith('/messages')){
  let citations=structuredClone(evidence);
  if(mode==='unsafe')citations[0].source_url='https://example.invalid/steal';
  if(mode==='legacy')citations=citations.map(c=>({n:c.n,question:c.question,answer:c.answer,section_path:c.section_path,source_file:'product-faq.md',document_sha256:'same',source_line:2}));
  const message={role:'assistant',content:body,...(mode==='legacy'?{sources:citations}:{citations,sources:citations})};
  res.setHeader('Content-Type','application/json');res.end(JSON.stringify({conversation_id:'101',messages:[message]}));return;
 }
 if(url.pathname.startsWith('/v1/knowledge/documents/')){
  documentCalls.push(url.pathname);
  if(mode==='missing'){res.statusCode=404;res.end();return;}
  res.setHeader('Content-Type','application/json');res.end(JSON.stringify({text,sha256:mode==='changed'?'different':'same'}));return;
 }
 if(url.pathname==='/v1/chat/stream'){
  res.setHeader('Content-Type','text/event-stream');
  res.write(event('session',{conversation_id:'101'})+event('retrieval_status',{state:'running',stage:'understanding'}));
  setTimeout(()=>{
   res.write(event('retrieval_status',{state:'running',stage:'generation'}));
   res.write(event('citations',{citations:evidence,message_id:'201'}));
   readyAt=Date.now();res.write(event('token',{text:body}));
   if(mode!=='interrupted')res.write(event('done',{conversation_id:'101',message_id:'201'}));
   res.end();
  },300);return;
 }
 res.statusCode=404;res.end();
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base='http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:850}});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(!r.url().startsWith(base))foreignCalls.push(r.url())});
  await page.goto(base);await page.locator('#messageInput').fill('运费如何计算？');await page.locator('#sendButton').click();
  await page.getByText('正在理解问题…',{exact:true}).waitFor();check('等待期间显示实际理解阶段',true);
  await page.locator('.citation').first().waitFor();
  const renderMs=Date.now()-readyAt;check('完整已验证正文及时渲染（无打字队列）',renderMs<1200);
  check('新回答支持非连续编号1/3/10',JSON.stringify(await page.locator('.citation').allTextContents())===JSON.stringify(['[1]','[3]','[10]']));
  check('未知编号保留纯文本',await page.locator('.bubble').last().textContent().then(t=>t.includes('[99]')));
  for(const n of [1,3,10]){
   await page.getByRole('button',{name:'查看来源 '+n,exact:true}).click();
   await page.locator('#fullDocument:not([hidden])').waitFor();
   check('来源'+n+'原文快照及章节正确',await page.locator('#sourceExcerpt').textContent()===evidence.find(c=>c.n===n).answer && (await page.locator('#sourcePath').textContent()).endsWith(String(n)));
   check('来源'+n+'高亮完整行范围',await page.locator('.source-line.highlight').count()===2);
   await page.locator('#sourceClose').click();
  }
  check('文档路径使用白名单文件名',documentCalls.every(u=>u==='/v1/knowledge/documents/product-faq.md'));
  check('原文作为文本安全展示',await page.locator('#sourceExcerpt script').count()===0);
  for(const variant of ['history','legacy','changed','missing','unsafe']){
   mode=variant;documentCalls=[];await page.reload();await page.locator('.citation').first().waitFor();
   const button=page.getByRole('button',{name:'查看来源 1',exact:true});await button.focus();await page.keyboard.press('Enter');
   await page.locator('#sourceDialog[open]').waitFor();
   if(variant==='unsafe'){check('拒绝外站来源URL并保留快照',documentCalls.length===0 && (await page.locator('#sourceExcerpt').textContent()).includes('原文快照1'));}
   else if(variant==='missing'){await page.getByText('完整文档暂时无法读取，上方知识片段仍可查看。',{exact:true}).waitFor();check('文件缺失仍显示原文快照',!(await page.locator('#sourceExcerpt').textContent()).includes('未保存'));}
   else {await page.locator('#fullDocument:not([hidden])').waitFor();check(variant+'历史/版本定位兼容',await page.locator('.highlight').count()===(variant==='changed'?0:variant==='legacy'?1:2));}
   await page.keyboard.press('Escape');
  }
  mode='history';await page.setViewportSize({width:390,height:844});await page.reload();await page.locator('.citation').first().click();await page.locator('#sourceDialog[open]').waitFor();
  check('窄屏来源弹窗可读且不横向越界',await page.locator('#sourceDialog').evaluate(e=>e.getBoundingClientRect().width<=innerWidth));await page.keyboard.press('Escape');
  mode='interrupted';await page.locator('#newChatTop').click();await page.locator('#messageInput').fill('中断测试');await page.locator('#sendButton').click();await page.getByText('连接提前中断，请重试。',{exact:true}).waitFor();
  check('中断响应不标记完成也不伪造可点击引用',await page.locator('.citation').count()===0);
  check('无JavaScript异常或外站请求',errors.length===0&&foreignCalls.length===0);
  const output={checks:checks.map(name=>({name,passed:true})),render_ms:renderMs,scope:'Actual HTML in Edge with authored HTTP/SSE fixtures; no customer writes or model requests.'};
  fs.writeFileSync(path.join(__dirname,'../eval/ch04/citations_ui_validation.json'),JSON.stringify(output,null,2)+'\n');
  console.log(JSON.stringify({passed:checks.length,render_ms:renderMs}));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1});
