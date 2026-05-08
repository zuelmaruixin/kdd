print("SCRIPT START", flush=True)

from pathlib import Path
import re
import json
import urllib.request
import urllib.error

cfg_path = Path("configs/router.deepseek.yaml")
print("config exists:", cfg_path.exists(), flush=True)

cfg = cfg_path.read_text()

keys = re.findall(r"api_key:\s*([^\s#]+)", cfg)
bases = re.findall(r"api_base:\s*([^\s#]+)", cfg)

print("found api_key count:", len(keys), flush=True)
print("found api_base count:", len(bases), flush=True)

if not keys:
    raise SystemExit("No api_key found")

if not bases:
    raise SystemExit("No api_base found")

key = keys[0].strip()
base = bases[0].strip().rstrip("/")

print("api_base:", base, flush=True)
print("api_key:", key[:6] + "..." + key[-4:] if len(key) > 10 else "***", flush=True)

if key == "REPLACE_WITH_YOUR_DEEPSEEK_KEY":
    raise SystemExit("API key is still placeholder")

url = base + "/chat/completions"
print("request url:", url, flush=True)

payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "max_tokens": 8,
    "temperature": 0
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

print("sending request...", flush=True)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        print("HTTP status:", resp.status, flush=True)
        print(body[:1000], flush=True)
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code, flush=True)
    print(e.read().decode("utf-8")[:2000], flush=True)
except Exception as e:
    print(type(e).__name__ + ":", e, flush=True)
