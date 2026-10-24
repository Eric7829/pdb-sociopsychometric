"""Record API responses from a persistent, human-accessible browser session."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

MAX_JSON_BYTES = 20 * 1024 * 1024
SECRET_KEYS = {'token', 'access_token', 'refresh_token', 'authorization', 'cookie', 'password', 'api_key'}


def jsonl_rows(path: Path) -> list:
    """Parse a JSONL file split on newline only: str.splitlines() would also break on U+2028/U+0085 that captured text
    (written with ensure_ascii=False) can contain."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def allowed_url(url: str) -> bool:
    parts = urlsplit(url)
    sensitive = any(part in {'oauth', 'auth', 'login', 'logout', 'register', 'send_digits', 'verify_digits'} for part in parts.path.split('/'))
    return (not sensitive and parts.scheme == 'https' and parts.hostname == 'api.personality-database.com'
            and parts.port in (None, 443) and parts.path.startswith(('/api/v1/', '/api/v2/')))


def sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, '[REDACTED]' if k.casefold() in SECRET_KEYS else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))


def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.casefold() in SECRET_KEYS else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


@dataclass
class Capture:
    method: str
    url: str
    request_body: dict | None
    status: int
    captured_at: str
    payload: dict | list
    sha256: str
    attempt: str | None = None


class Recorder:
    def __init__(self, directory: Path, max_bytes: int = MAX_JSON_BYTES):
        self.directory = directory
        self.max_bytes = max_bytes
        self.captures: list[Capture] = []
        self.errors: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)

    def record(self, method, url, request_body, status, payload, attempt=None) -> Capture:
        if not allowed_url(url):
            raise ValueError('Not an allowed PDB API URL')
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        if len(encoded) > self.max_bytes:
            raise ValueError('JSON exceeds capture size limit')
        payload = redact(payload)
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        capture = Capture(method, sanitize_url(url), redact(request_body), status,
                          datetime.now(timezone.utc).isoformat(), payload, digest, attempt)
        with (self.directory / 'pdb.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(asdict(capture), ensure_ascii=False) + '\n')
        self.captures.append(capture)
        return capture

    async def response(self, response):
        if not allowed_url(response.url):
            return
        try:
            if int(response.headers.get('content-length', '0')) > self.max_bytes:
                raise ValueError('JSON exceeds capture size limit')
            if 'json' not in response.headers.get('content-type', '').lower():
                raise ValueError(f'Non-JSON API response: HTTP {response.status}')
            body = await response.body()
            if len(body) > self.max_bytes:
                raise ValueError('JSON exceeds capture size limit')
            payload = json.loads(body)
            if not isinstance(payload, (dict, list)):
                raise ValueError('Expected JSON object or array')
            request_body = None
            if response.request.post_data:
                try:
                    request_body = response.request.post_data_json
                except Exception:
                    request_body = {'unparsed_body': '[OMITTED]'}
            self.record(response.request.method, response.url, request_body, response.status, payload)
        except Exception as error:
            self.errors.append(f'{sanitize_url(response.url)}: {type(error).__name__}: {error}')


async def finalize_capture(recorder, page, navigation_error):
    closed = page.is_closed()
    summary = {'url': page.url, 'title': None, 'captures': len(recorder.captures),
               'errors': list(recorder.errors), 'navigation_error': navigation_error,'browser_closed':closed}
    if not closed:
        try:
            summary['title']=await page.title()
            await page.screenshot(path=str(recorder.directory/'page.png'))
        except Exception as error:
            summary['errors'].append(f'Browser summary: {type(error).__name__}')
    (recorder.directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


async def capture_session(start_url: str, output_dir: Path, user_data_dir: Path,
                          *, seconds: float = 30, headless: bool = False) -> list[Capture]:
    from playwright.async_api import async_playwright, Error as PlaywrightError
    recorder = Recorder(output_dir)
    pending: set[asyncio.Task] = set()
    navigation_error = None
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(str(user_data_dir.resolve()), headless=headless)
        def schedule(response):
            task = asyncio.create_task(recorder.response(response))
            pending.add(task)
            task.add_done_callback(pending.discard)
        context.on('response', schedule)
        page = context.pages[0] if context.pages else await context.new_page()
        try:
            try:
                await page.goto(start_url, wait_until='domcontentloaded', timeout=60000)
            except Exception as error:
                navigation_error = f'{type(error).__name__}: {error}'
            deadline = asyncio.get_running_loop().time() + seconds
            while asyncio.get_running_loop().time() < deadline:
                if page.is_closed(): break
                try:
                    await page.mouse.wheel(0, 1200)
                except PlaywrightError:
                    if page.is_closed(): break
                    raise
                await asyncio.sleep(min(2, max(0, deadline - asyncio.get_running_loop().time())))
        finally:
            context.remove_listener('response', schedule)
            if pending:
                await asyncio.gather(*pending)
            try:
                await finalize_capture(recorder,page,navigation_error)
            finally:
                await context.close()
    return recorder.captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, default=Path('data/captures'))
    parser.add_argument('--session', type=Path, default=Path('data/browser'))
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    captures = asyncio.run(capture_session(args.url, args.output, args.session, seconds=args.seconds, headless=args.headless))
    print(json.dumps({'captures': len(captures), 'summary': str(args.output / 'summary.json')}))
    return 0 if captures else 1


if __name__ == '__main__':
    raise SystemExit(main())
