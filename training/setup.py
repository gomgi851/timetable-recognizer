"""Download only official public pretrained weights; training images remain local."""
from pathlib import Path
import hashlib, json, urllib.request, zipfile

root=Path('training/private'); root.mkdir(parents=True,exist_ok=True)
url='https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip'
dest=root/'english_g2.pth'
if not dest.exists():
    archive=root/'english_g2.zip'
    urllib.request.urlretrieve(url,archive)
    with zipfile.ZipFile(archive) as z:
        with z.open('english_g2.pth') as source:
            dest.write_bytes(source.read())
expected='5864788e1821be9e454ec108d61b887d'
assert hashlib.md5(dest.read_bytes()).hexdigest()==expected, 'Official model checksum mismatch'
metadata={'source':url,'officialMD5':expected,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size}
(root/'pretrained-source.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata),flush=True)
