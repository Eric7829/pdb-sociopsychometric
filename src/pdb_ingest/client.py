from urllib.parse import urlencode
from .adapters import parse_listing,parse_counts,parse_comment
from .models import Page,SchemaError

class PdbClient:
    def __init__(self,transport,*,comment_limit=100,sort=None):
        self.transport=transport
        self.sort=sort  # None = site default (HOT); 'top' = Top / All Time
        self.comment_limit=comment_limit
        self.details={}

    async def listing(self,token):
        offset=int(token or 0)
        query=urlencode(dict(offset=offset,limit=100,cid=15,pid=1,sub_cat_id=4065,**({'sort':self.sort} if self.sort else {}),cat_id=15,property_id=1))
        capture=await self.transport.get('/api/v1/profiles?'+query)
        return parse_listing(capture.payload,offset,100)

    async def detail(self,profile_num):
        self.details[profile_num]=await self.transport.get(f'/api/v1/profile/{profile_num}')
        return self.details[profile_num]

    async def tallies(self,profile_num):
        if profile_num not in self.details: await self.detail(profile_num)
        return parse_counts(self.details[profile_num].payload)

    @staticmethod
    def comment_page(payload):
        if not isinstance(payload.get('comments'),list) or type(payload.get('has_more')) is not bool:
            raise SchemaError('Missing comment pagination fields')
        more=payload['has_more']
        if more and type(payload.get('next_offset')) is not int:
            raise SchemaError('Missing comment next_offset')
        return Page([parse_comment(r) for r in payload['comments']],str(payload['next_offset']) if more else None,not more)

    async def comments(self,profile_num,token):
        query=urlencode(dict(sort='NEW',offset=int(token or 0),range='all',limit=self.comment_limit,version='W3'))
        return self.comment_page((await self.transport.get(f'/api/v1/comments/{profile_num}?'+query)).payload)

    async def replies(self,profile_num,comment_id,token):
        query=urlencode(dict(limit=100,offset=int(token or 0),comment_id=comment_id))
        return self.comment_page((await self.transport.get('/api/v1/comment_replies?'+query)).payload)
