"""Bounded API reconnaissance; no credentials are written to disk."""
import asyncio
import json
import hashlib
from pathlib import Path
from playwright.async_api import async_playwright
from pdb_ingest.capture import Recorder, allowed_url

async def main():
    recorder=Recorder(Path('data/discovery/probe'))
    tasks=set()
    auth={}
    async with async_playwright() as p:
        context=await p.chromium.launch_persistent_context(str(Path('data/browser').resolve()),headless=False)
        async def handle(response):
            if response.request.resource_type=='script' and response.url.startswith('https://static2.personalitydatabase.net/'):
                directory=Path('data/discovery/scripts'); directory.mkdir(exist_ok=True)
                try:
                    body=await response.body()
                    (directory/(hashlib.sha256(response.url.encode()).hexdigest()[:16]+'.js')).write_bytes(body)
                except Exception: pass
            if allowed_url(response.url):
                headers=await response.request.all_headers()
                for key,value in headers.items():
                    if key.startswith('x-') or key=='authorization': auth[key]=value
                await recorder.response(response)
        def schedule(response):
            task=asyncio.create_task(handle(response)); tasks.add(task); task.add_done_callback(tasks.discard)
        context.on('response',schedule)
        page=context.pages[0]
        await page.goto('https://www.personality-database.com/profile/1912391/esqueleto-narrativo-pdb-profiles-mbti-personality-type',wait_until='domcontentloaded')
        print('TITLE',await page.title(),flush=True)
        if 'Just a moment' in await page.title():
            print('CHALLENGE: waiting for manual browser verification',flush=True)
            await page.wait_for_function("!document.title.includes('Just a moment')",timeout=180000)
        await page.wait_for_timeout(5000)
        await page.mouse.wheel(0,1600)
        await page.wait_for_timeout(4000)
        print('BUTTONS',json.dumps(await page.get_by_role('button').all_text_contents()))
        print('REPLIES',json.dumps(await page.get_by_text('replies',exact=False).all_text_contents())[:2500])
        print('AUTH_HEADER_NAMES',list(auth))
        for locator in (page.get_by_text('View 1 reply',exact=False),page.get_by_text('View 6 replies',exact=False),page.get_by_text('View replies',exact=False)):
            if await locator.count():
                await locator.first.click(); await page.wait_for_timeout(2500)
        # Verify a second actual listing page through the same browser context.
        listing_url='https://api.personality-database.com/api/v1/profiles?offset=100&limit=100&cid=15&pid=1&sub_cat_id=4065&cat_id=15&property_id=1'
        data=await page.evaluate('''async ({url,headers})=>{const r=await fetch(url,{headers,credentials:'include'});return {status:r.status,body:await r.text()}}''',{'url':listing_url,'headers':auth})
        print('LISTING2',data['status'],data['body'][:100])
        for path in ('/api/v1/comment_replies?limit=10&offset=0&comment_id=13164021', '/api/v1/comment_replies?limit=10&offset=10&comment_id=13164021', '/api/v1/comments/1912391?sort=NEW&offset=1400&range=all&limit=20&version=W3'):
            await page.evaluate('''async url=>{await fetch(url,{credentials:'include'})}''','https://api.personality-database.com'+path)
            await page.wait_for_timeout(2000)
        if tasks: await asyncio.gather(*tasks)
        print('CAPTURES',json.dumps([c.url for c in recorder.captures]))
        await context.close()

asyncio.run(main())
