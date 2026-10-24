"""OpenRouter chat completions constrained to one forced tool call per request."""
import asyncio
import json
import os
from pathlib import Path
import httpx

URL='https://openrouter.ai/api/v1/chat/completions'
DEFAULT_MODEL='deepseek/deepseek-v4-flash'
# Verified live: fp8 weights, honours forced tool_choice and seed, and its prefix cache hits
# (2nd call: 11,520 of 11,620 prompt tokens cached, -77% cost). StreamLake/Baidu/GMICloud reject tool_choice.
DEFAULT_PROVIDER='deepinfra/fp8'

class LlmError(RuntimeError): pass
class ToolCallError(LlmError): pass

def load_api_key(path=Path('OPEN_ROUTER_API_KEY.txt')):
    """Environment variable first, then the local key file. The key is never logged or stored."""
    key=os.environ.get('OPEN_ROUTER_API_KEY') or (path.read_text(encoding='utf-8').strip() if path.exists() else '')
    if not key: raise LlmError(f'No OpenRouter key: set OPEN_ROUTER_API_KEY or create {path}')
    return key

class OpenRouter:
    def __init__(self,http,api_key,model=DEFAULT_MODEL,*,provider=None,max_tokens=16000,sleep=asyncio.sleep):
        self.http=http; self.api_key=api_key; self.model=model; self.max_tokens=max_tokens; self.sleep=sleep
        # Pinning one upstream keeps weights/quantization constant across calls and lets prefix caching hit.
        self.provider={'require_parameters':True,**({'order':[provider],'allow_fallbacks':False} if provider else {})}

    async def call_tool(self,messages,tool,*,temperature,seed):
        name=tool['function']['name']
        body={'model':self.model,'messages':messages,'tools':[tool],
              'tool_choice':{'type':'function','function':{'name':name}},'temperature':temperature,'seed':seed,
              'max_tokens':self.max_tokens,'usage':{'include':True},'provider':self.provider,
              # Native thinking off: DeepInfra's default already is, and 'effort: low' would turn it ON (+1.2k tokens/call).
              # The predict style asks for its reasoning explicitly in the tool arguments instead.
              'reasoning':{'enabled':False}}
        # 8 attempts: provider 429s come in bursts; waits honour Retry-After and grow to ~2 min in total before giving up.
        for attempt in range(8):
            try:
                response=await self.http.post(URL,headers={'Authorization':'Bearer '+self.api_key},json=body,timeout=300)
            except httpx.HTTPError as error:
                if attempt==7: raise LlmError(f'OpenRouter transport: {type(error).__name__}') from error
                await self.sleep(2**(attempt+1)); continue
            if response.status_code==429 or response.status_code>=500:
                if attempt==7: raise LlmError(f'OpenRouter retry limit: HTTP {response.status_code}')
                try: wait=float(response.headers.get('retry-after',''))
                except ValueError: wait=0
                await self.sleep(min(60,max(wait,2**(attempt+1)))); continue
            if response.status_code!=200: raise LlmError(f'OpenRouter HTTP {response.status_code}: {response.text[:300]}')
            data=response.json()
            if 'error' in data: raise LlmError(f"OpenRouter error: {json.dumps(data['error'])[:300]}")
            choice=data['choices'][0]; message=choice['message']
            result={'id':data.get('id'),'provider':data.get('provider'),'usage':data.get('usage') or {},
                    'finish_reason':choice.get('finish_reason'),'content':message.get('content'),
                    'reasoning':message.get('reasoning'),'arguments':None}
            calls=[c for c in message.get('tool_calls') or [] if c.get('function',{}).get('name')==name]
            if not calls: raise ToolCallError('Model returned no tool call',result)
            try:
                result['arguments']=json.loads(calls[0]['function']['arguments'])
            except json.JSONDecodeError as error:
                raise ToolCallError('Tool arguments are not JSON',result) from error
            return result
