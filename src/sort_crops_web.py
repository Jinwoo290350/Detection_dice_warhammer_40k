"""Web-based crop sorter — runs a tiny local HTTP server. Open the URL in
your browser, then press keys 1..6 to file each crop into the corresponding
face folder. SPACE skips, D deletes the crop entirely (junk/duplicate).

Built on Python's stdlib http.server — no extra dependencies.

Usage:
    python src/sort_crops_web.py
    # then open http://localhost:8765 in any browser
"""

from __future__ import annotations

import json
import shutil
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "classifier_dataset"
UNSORTED = DATASET_DIR / "_unsorted"
PORT = 8765

INDEX_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>Sort dice crops</title>
<style>
 body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#eee;
      margin:0;padding:24px;text-align:center}
 #counts{font-size:18px;margin-bottom:12px}
 #counts span{margin:0 8px}
 #img{max-width:90vw;max-height:60vh;border:2px solid #444;background:#000;
      image-rendering:pixelated}
 #name{font-size:11px;color:#888;margin-top:8px;word-break:break-all}
 #queue{font-size:13px;color:#aaa;margin-top:6px}
 .btnrow{margin-top:16px}
 button{font-size:18px;padding:10px 20px;margin:4px;border-radius:6px;
        border:1px solid #555;background:#333;color:#eee;cursor:pointer}
 button:hover{background:#555}
 .face{background:#2a4a2a}
 .skip{background:#3a3a5a}
 .del{background:#5a2a2a}
 kbd{background:#444;padding:2px 6px;border-radius:4px;font-size:11px}
</style></head>
<body>
<div id='counts'></div>
<img id='img' src='' alt='loading...'/>
<div id='name'></div>
<div id='queue'></div>
<div class='btnrow'>
  <button class='face' onclick="act('1')">1 <kbd>1</kbd></button>
  <button class='face' onclick="act('2')">2 <kbd>2</kbd></button>
  <button class='face' onclick="act('3')">3 <kbd>3</kbd></button>
  <button class='face' onclick="act('4')">4 <kbd>4</kbd></button>
  <button class='face' onclick="act('5')">5 <kbd>5</kbd></button>
  <button class='face' onclick="act('6')">6 <kbd>6</kbd></button>
  <button class='skip' onclick="act('skip')">skip <kbd>SPACE</kbd></button>
  <button class='del' onclick="act('delete')">delete <kbd>D</kbd></button>
</div>
<script>
async function refresh(){
  const r = await fetch('/state');
  const s = await r.json();
  const cs = Object.entries(s.counts).map(([k,v])=>`<span>${k}: ${v}</span>`).join('');
  document.getElementById('counts').innerHTML = cs + `<span>total: ${s.total_classified}</span>`;
  if(!s.current){
    document.getElementById('img').src='';
    document.getElementById('img').alt='ALL DONE';
    document.getElementById('name').textContent='';
    document.getElementById('queue').textContent=`queue empty (${s.processed} done)`;
    return;
  }
  document.getElementById('img').src='/crop?path='+encodeURIComponent(s.current);
  document.getElementById('name').textContent=s.current;
  document.getElementById('queue').textContent=`${s.position}/${s.queue_size} in queue`;
}
async function act(a){
  await fetch('/action?a='+a, {method:'POST'});
  refresh();
}
document.addEventListener('keydown', e=>{
  if('123456'.includes(e.key)) act(e.key);
  else if(e.key===' '){e.preventDefault(); act('skip');}
  else if(e.key.toLowerCase()==='d') act('delete');
});
refresh();
</script></body></html>
"""


class State:
    def __init__(self) -> None:
        self.queue: list[Path] = sorted(UNSORTED.glob("*.jpg"))
        self.idx = 0
        for f in range(1, 7):
            (DATASET_DIR / str(f)).mkdir(exist_ok=True)

    def current(self) -> Path | None:
        return self.queue[self.idx] if self.idx < len(self.queue) else None

    def counts(self) -> dict[str, int]:
        return {str(f): len(list((DATASET_DIR / str(f)).glob("*.jpg"))) for f in range(1, 7)}

    def state(self) -> dict:
        cur = self.current()
        return {
            "current": cur.name if cur else None,
            "position": self.idx + 1 if cur else self.idx,
            "queue_size": len(self.queue),
            "processed": self.idx,
            "counts": self.counts(),
            "total_classified": sum(self.counts().values()),
        }

    def act(self, action: str) -> None:
        cur = self.current()
        if cur is None:
            return
        if action in {"1", "2", "3", "4", "5", "6"}:
            dst = DATASET_DIR / action / cur.name
            if cur.exists():
                shutil.move(str(cur), str(dst))
        elif action == "delete":
            if cur.exists():
                cur.unlink()
        # skip / unknown → just advance
        self.idx += 1


STATE = State()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 — silence stdlib log spam
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — stdlib API
        if self.path == "/" or self.path == "/index.html":
            self._send(200, INDEX_HTML.encode("utf-8"))
        elif self.path == "/state":
            self._send(200, json.dumps(STATE.state()).encode("utf-8"), "application/json")
        elif self.path.startswith("/crop?"):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            name = params.get("path", [""])[0]
            path = UNSORTED / name
            if not path.exists() or not path.is_file():
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, path.read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/action?"):
            qs = urllib.parse.urlparse(self.path).query
            action = urllib.parse.parse_qs(qs).get("a", [""])[0]
            STATE.act(action)
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def main() -> None:
    if not UNSORTED.exists() or not any(UNSORTED.glob("*.jpg")):
        raise SystemExit(f"no crops in {UNSORTED}")
    print(f"sort tool — open http://localhost:{PORT}")
    print("keys: 1..6 = move to face folder, SPACE = skip, D = delete, ctrl-C = stop")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
