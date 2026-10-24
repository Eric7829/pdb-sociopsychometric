import asyncio
import re
from urllib.parse import urlsplit, urljoin
import httpx
from .models import Document

GDOC_REGEX = re.compile(r'https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)')

def extract_documents(comments):
    keys = {}
    for comment in comments:
        for key in GDOC_REGEX.findall(comment.text):
            refs = keys.setdefault(key, [])
            if comment.comment_id not in refs:
                refs.append(comment.comment_id)
    return keys

def allowed_redirect(url):
    parts = urlsplit(url)
    host = parts.hostname or ''
    return parts.scheme == 'https' and parts.port in (None,443) and (
        host in {'docs.google.com', 'drive.google.com'} or host.endswith('.googleusercontent.com'))

async def fetch_document(key, source_comment_ids, client, *, max_bytes=5*1024*1024, sleep=asyncio.sleep):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
        raise ValueError('Invalid Google document key')
    def result(status, text=None, error=None):
        return Document(key, source_comment_ids, status, text, error)
    for attempt in range(3):
        url = f'https://docs.google.com/document/d/{key}/export?format=txt'
        try:
            for redirect in range(6):
                async with client.stream('GET', url, follow_redirects=False, timeout=30) as response:
                    if response.status_code in (401,403,404,410):
                        return result('UNAVAILABLE', error=f'HTTP {response.status_code}')
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get('location', ''))
                        if not allowed_redirect(url):
                            return result('UNAVAILABLE', error='Non-export redirect')
                        continue
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt == 2:
                            return result('ERROR', error=f'HTTP {response.status_code}')
                        delay = response.headers.get('retry-after', '')
                        await sleep(max(2 ** (attempt+1), float(delay) if delay.isdigit() else 0))
                        break
                    if response.status_code != 200:
                        return result('ERROR', error=f'HTTP {response.status_code}')
                    if 'text/plain' not in response.headers.get('content-type', '').lower():
                        return result('UNAVAILABLE', error='Export was not plaintext')
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            return result('ERROR', error='Export exceeds size limit')
                    text = body.decode('utf-8-sig')
                    if re.match(r'\s*(?:<!doctype\s+html|<html)', text, re.I):
                        return result('UNAVAILABLE', error='HTML login/challenge response')
                    return result('AVAILABLE', text)
            else:
                return result('UNAVAILABLE', error='Redirect limit')
        except (httpx.HTTPError, UnicodeDecodeError) as error:
            if attempt == 2:
                return result('ERROR', error=type(error).__name__)
            await sleep(2 ** (attempt+1))
    return result('ERROR', error='Retry limit')
