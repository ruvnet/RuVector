"""Private persistent adapter artifacts, addressed only by random opaque handles."""
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from fastapi import HTTPException


class AdapterStore:
    def __init__(self, root, max_adapters=128):
        self.max_adapters = max_adapters
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_mode & 0o077:
            raise ValueError('Adapter root must be a private nonsymlink directory')

    def allocate(self, account="local-test"):
        owner = hashlib.sha256(account.encode()).hexdigest()
        # Fail closed at bounded capacity. Expired/incomplete state is retained for
        # explicit operator maintenance; never delete runtime state automatically.
        if sum(1 for p in self.root.iterdir() if p.is_dir()) >= self.max_adapters:
            raise HTTPException(503, 'Adapter capacity reached; operator maintenance required')
        owned = sum(1 for p in self.root.iterdir() if p.is_dir()
                    and (p / 'owner').exists() and (p / 'owner').read_text() == owner)
        if owned >= 4:
            raise HTTPException(503, 'Tenant adapter capacity reached; operator maintenance required')
        handle = secrets.token_hex(32)
        directory = self.root / handle
        directory.mkdir(mode=0o700)
        marker = directory / 'owner'
        with marker.open('x') as out:
            os.chmod(marker, 0o600)
            out.write(owner)
        return handle, directory

    def publish(self, directory, metadata):
        digest = hashlib.sha256()
        weights = sorted(directory.rglob('*.safetensors'))
        if not weights:
            raise ValueError('No trained adapter weights')
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                path.chmod(0o600)
            elif path.is_dir():
                path.chmod(0o700)
        for path in weights:
            digest.update(path.read_bytes())
        metadata['adapter_fingerprint'] = digest.hexdigest()
        temporary = directory / 'metadata.tmp'
        with temporary.open('x') as out:
            os.chmod(temporary, 0o600)
            json.dump(metadata, out)
        temporary.replace(directory / 'metadata.json')
        return metadata

    def resolve(self, handle, account, host, model):
        if not isinstance(handle, str) or not re.fullmatch('[0-9a-f]{64}', handle):
            raise HTTPException(404, 'Unknown adapter')
        directory = self.root / handle
        try:
            if directory.is_symlink():
                raise ValueError('symlink')
            metadata = json.loads((directory / 'metadata.json').read_text())
            if (metadata['account_id'], metadata['host'], metadata['model']) != (account, host, model):
                raise ValueError('binding')
            if metadata['expires_at'] <= time.time():
                raise ValueError('expired')
            digest = hashlib.sha256()
            for path in sorted(directory.rglob('*.safetensors')):
                if path.is_symlink():
                    raise ValueError('symlink')
                digest.update(path.read_bytes())
            if digest.hexdigest() != metadata['adapter_fingerprint']:
                raise ValueError('artifact integrity')
            return directory, metadata
        except Exception as exc:
            raise HTTPException(404, 'Unknown or unavailable adapter') from exc
