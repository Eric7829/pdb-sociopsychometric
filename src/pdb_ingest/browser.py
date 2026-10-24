"""Use Chromium's own fetch stack, cookies and TLS for PDB JSON."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from .models import AccessBlocked

class BrowserResponse:
    def __init__(self,data):
        self.status=data['status']; self.headers=data['headers']; self.text=data['text']
    async def body(self): return self.text.encode('utf-8')
    async def dispose(self): pass

class BrowserRequest:
    def __init__(self,page): self.page=page
    async def get(self,url,*,headers=None,timeout=30000):
        try:
            result=await self.page.evaluate('''async ({url,headers,timeout})=>{
                const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),timeout);
                try {
                    const r=await fetch(url,{headers,credentials:'include',signal:controller.signal});
                    const reader=r.body.getReader();const chunks=[];let size=0;
                    while(true){const part=await reader.read();if(part.done)break;
                        size+=part.value.length;if(size>20*1024*1024){await reader.cancel();throw new Error('JSON size limit');}chunks.push(part.value);}
                    const body=new Uint8Array(size);let offset=0;for(const chunk of chunks){body.set(chunk,offset);offset+=chunk.length;}
                    return {status:r.status,headers:Object.fromEntries(r.headers.entries()),text:new TextDecoder().decode(body)};
                }finally{clearTimeout(timer);}
            }''',{'url':url,'headers':headers or {},'timeout':timeout})
            return BrowserResponse(result)
        except Exception as error:
            if 'size limit' in str(error): raise ValueError('JSON size limit') from error
            if 'Target' in str(error): raise AccessBlocked('Browser session closed') from error
            raise OSError('Browser fetch failed; session/network may need attention') from error

@asynccontextmanager
async def browser_session(session_path: Path,*,headless=False,login_seconds=120):
    async with async_playwright() as p:
        context=await p.chromium.launch_persistent_context(str(session_path.resolve()),headless=headless)
        try:
            page=await context.new_page()
            await page.goto('https://www.personality-database.com/profile?pid=1&cid=15&sub_cat_id=4065',wait_until='domcontentloaded',timeout=60000)
            if 'Just a moment' in await page.title():
                print('Cloudflare verification: complete the check in the browser.',flush=True)
                try:
                    await page.wait_for_function("!document.title.includes('Just a moment')",timeout=login_seconds*1000)
                except PlaywrightTimeout as error:
                    raise AccessBlocked('Cloudflare verification not completed') from error
            yield BrowserRequest(page)
        finally:
            await context.close()
