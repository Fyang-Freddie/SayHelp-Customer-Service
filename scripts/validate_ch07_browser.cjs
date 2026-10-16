// Native page in real Edge, with deterministic HTTP fixtures and no paid calls.
const fs=require('fs'),http=require('http'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
let ids=['2','1'],failList=false,holdA=false,releaseA,chatCalls=0;
const checks=[];
const json=(res,data,status=200)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(data));};
const event=(name,data)=>`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');
 if(url.pathname==='/'){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(fs.readFileSync(path.join(__dirname,'../app/web/index.html')));return;}
 if(['/api/conversations','/v1/conversations'].includes(url.pathname)){
  return json(res,{conversations:ids.map(id=>({id,title:'首问'+id,has_summary:id==='1',is_pinned:false,updated_at:'2026-10-16T02:00:00'})),next_cursor:null},failList?503:200);
 }
 const match=url.pathname.match(/^\/(?:api|v1)\/conversations\/(\d+)\/messages$/);
 if(match){const finish=()=>json(res,{conversation_id:match[1],messages:[{id:match[1]+'1',role:'user',content:'用户完整原话'+match[1]},{id:match[1]+'2',role:'assistant',content:'客服完整原话'+match[1],actions:[]}]});if(holdA&&match[1]==='1')releaseA=finish;else finish();return;}
 if(url.pathname==='/v1/chat/stream'){
  let body='';req.on('data',part=>body+=part);req.on('end',()=>{
   chatCalls++;const request=JSON.parse(body),id=request.conversation_id||String(ids.length+1);
   if(!ids.includes(id))ids.unshift(id);
   res.writeHead(200,{'Content-Type':'text/event-stream'});
   res.end(event('session',{conversation_id:id})+event('token',{text:'继续聊成功'+chatCalls})+event('done',{conversation_id:id,message_id:id+'3'}));
  });return;
 }
 res.writeHead(404);res.end();
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:850}});
  page.setDefaultTimeout(5000);
  await page.goto('http://127.0.0.1:'+server.address().port);
  await page.getByText('客服完整原话2',{exact:true}).waitFor();
  assert.equal(await page.locator('[data-conversation="1"]').getByText('已摘要',{exact:true}).count(),1);
  checks.push('首问预览及已摘要标记');
  await page.locator('[data-conversation="1"]').click();
  await page.getByText('用户完整原话1',{exact:true}).waitFor();
  await page.getByText('客服完整原话1',{exact:true}).waitFor();
  checks.push('切回旧会话回载完整原文');
  holdA=true;
  await page.locator('[data-conversation="1"]').click();
  for(let i=0;!releaseA&&i<50;i++)await new Promise(resolve=>setTimeout(resolve,10));
  assert.ok(releaseA);
  await page.locator('[data-conversation="2"]').click();
  await page.getByText('客服完整原话2',{exact:true}).waitFor();
  releaseA();holdA=false;
  await page.waitForTimeout(100);
  assert.equal(await page.getByText('客服完整原话1',{exact:true}).count(),0);
  checks.push('A晚返回不会覆盖已切换的B');
  await page.locator('#newChatTop').click();
  assert.equal(await page.locator('[data-conversation="1"]').count(),1);
  await page.locator('#messageInput').fill('全新会话');await page.locator('#sendButton').click();
  await page.getByText('继续聊成功1',{exact:true}).waitFor();
  await page.locator('[data-conversation="3"]').waitFor();
  checks.push('新对话保留旧会话并追加侧栏');
  failList=true;
  await page.locator('#messageInput').fill('侧栏失败仍能聊');await page.locator('#sendButton').click();
  await page.getByText('继续聊成功2',{exact:true}).waitFor();
  await page.waitForTimeout(100);
  assert.ok(!(await page.locator('body').innerText()).includes('历史暂时无法加载'));
  checks.push('侧栏加载失败静默且聊天可用');
  console.log(JSON.stringify({browser:'Edge',checks,paid_calls:0},null,2));
 }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
