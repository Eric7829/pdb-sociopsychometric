import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from .capture import Recorder, allowed_url
from .models import AccessBlocked, ProfileUnavailable, SchemaError, profile_refusal

class JsonTransport:
    def __init__(self, request, capture_dir, *, headers=None, sleep=asyncio.sleep, clock=time.monotonic, interval=1):
        self.request=request
        self.recorder=Recorder(capture_dir)
        self.headers=headers or {}
        self.sleep=sleep
        self.clock=clock
        self.last_request=None
        self.interval=interval  # minimum seconds between PDB requests
        self.attempt=None

    async def get(self, path):
        url = path if path.startswith('https://') else 'https://api.personality-database.com'+path
        if not allowed_url(url): raise ValueError('Disallowed API endpoint')
        for attempt in range(3):
            if self.last_request is not None:
                await self.sleep(max(0,self.interval-(self.clock()-self.last_request)))
            self.last_request=self.clock()
            response=None
            try:
                response=await self.request.get(url,headers=self.headers,timeout=30000)
                if response.status==403 and 'json' in response.headers.get('content-type',''):
                    try: payload=json.loads(await response.body())
                    except ValueError: payload=None
                    if profile_refusal(url,payload):
                        self.recorder.record('GET',url,None,response.status,payload,self.attempt)
                        raise ProfileUnavailable(payload['message'])
                    if isinstance(payload,dict) and payload.get('message'):
                        raise AccessBlocked(f"HTTP 403: {str(payload['message'])[:200]}")
                if response.status in (401,403): raise AccessBlocked(f'HTTP {response.status}: login or challenge required')
                if response.status==429 or response.status>=500:
                    if attempt==2: raise RuntimeError(f'API retry limit: HTTP {response.status}')
                    delay=response.headers.get('retry-after','')
                    try:
                        retry=float(delay)
                    except ValueError:
                        try: retry=max(0,(parsedate_to_datetime(delay)-datetime.now(timezone.utc)).total_seconds())
                        except (TypeError,ValueError): retry=0
                    await self.sleep(max(2**(attempt+1),retry))
                    continue
                if response.status != 200: raise RuntimeError(f'API HTTP {response.status}')
                if 'json' not in response.headers.get('content-type',''):
                    raise AccessBlocked('Non-JSON API response; possible challenge')
                body=await response.body()
                if len(body)>self.recorder.max_bytes: raise SchemaError('JSON size limit')
                payload=json.loads(body)
                if not isinstance(payload,(dict,list)): raise SchemaError('Expected JSON object/array')
                if profile_refusal(url,payload):
                    self.recorder.record('GET',url,None,response.status,payload,self.attempt)
                    raise ProfileUnavailable(payload['message'])
                if isinstance(payload,dict) and (payload.get('code') in (401,403) or payload.get('error',{}).get('code') in ('S40100','S40300')):
                    raise AccessBlocked(payload.get('message','Login required'))
                return self.recorder.record('GET',url,None,response.status,payload,self.attempt)
            except (TimeoutError,OSError) as error:
                if attempt==2: raise RuntimeError('API transport retry limit') from error
                await self.sleep(2**(attempt+1))
            finally:
                if response is not None: await response.dispose()
