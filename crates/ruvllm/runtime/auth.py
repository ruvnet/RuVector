"""Identity resource-token authentication; no alternate credential path."""
import os
import time
import jwt
from fastapi import HTTPException


class IdentityAuth:
    def __init__(self):
        self.issuer = os.environ['IDENTITY_ISSUER']
        self.actor = os.environ['IDENTITY_EXPECTED_ACTOR']
        url = os.environ['IDENTITY_JWKS_URL']
        if not url.startswith('https://'):
            raise ValueError('Identity JWKS must use HTTPS')
        self.keys = jwt.PyJWKClient(url, cache_jwk_set=True, lifespan=60)

    def verify(self, authorization):
        try:
            if not authorization or not authorization.startswith('Bearer '):
                raise ValueError('bearer required')
            token = authorization[7:]
            key = self.keys.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, key.key, algorithms=['ES256'],
                                audience='meta-proxy', issuer=self.issuer,
                                options={'require': ['exp', 'iat', 'sub', 'account_id', 'aud', 'iss']})
            if (claims.get('typ') != 'access' or claims.get('setup') is True
                    or claims.get('workload') is True or claims.get('exchanged') is not True
                    or claims.get('act') != self.actor
                    or claims.get('scope') != 'platform:microlora:run'
                    or claims['aud'] != 'meta-proxy'
                    or not isinstance(claims['account_id'], str) or not claims['account_id'].strip()
                    or not isinstance(claims['sub'], str) or not claims['sub'].strip()):
                raise ValueError('invalid resource authority')
            if any(isinstance(claims[k], bool) or not isinstance(claims[k], (int, float))
                   for k in ('iat', 'exp')):
                raise ValueError('invalid lifetime')
            if 'nbf' in claims and (isinstance(claims['nbf'], bool) or not isinstance(claims['nbf'], (int, float))):
                raise ValueError('invalid not-before')
            if not 0 < claims['exp'] - claims['iat'] <= 300 or claims['iat'] > time.time():
                raise ValueError('invalid lifetime')
            return claims
        except Exception as exc:
            raise HTTPException(401, 'Invalid Identity resource credential') from exc
