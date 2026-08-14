from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from manager_api.config.settings import ServerConfig, load_config


logger = logging.getLogger(__name__)


class ManagerWeb:
    def __init__(self, config: ServerConfig | None = None):
        self.config = config or load_config()
        self.httpd: ThreadingHTTPServer | None = None

    def start(self) -> None:
        handler = self._handler_class()
        self.httpd = ThreadingHTTPServer((self.config.host, self.config.manager_web_port), handler)
        logger.info("manager-web listening on %s:%s", self.config.host, self.config.manager_web_port)
        self.httpd.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.start, name="manager-web", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        web = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CruiseCarManagerWeb/0.1"

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path == "/health":
                    self._send_text("ok\n", "text/plain; charset=utf-8")
                elif path == "/":
                    self._send_text(web.index_html(), "text/html; charset=utf-8")
                elif path in {"/send", "/web_send"}:
                    self.send_response(HTTPStatus.FOUND.value)
                    self.send_header("Location", path + "/")
                    self.end_headers()
                elif path in {"/send/", "/web_send/"}:
                    self._send_text(web.web_sender_html(), "text/html; charset=utf-8")
                else:
                    self._send_text("not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.info("%s - %s", self.address_string(), fmt % args)

            def _send_text(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
                data = text.encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def web_sender_html(self) -> str:
        index_path = Path(__file__).resolve().parents[2] / "web_send" / "index.html"
        if not index_path.exists():
            return "<!doctype html><meta charset='utf-8'><title>web_send not found</title><body>web_send/index.html not found</body>"
        return index_path.read_text(encoding="utf-8")

    def index_html(self) -> str:
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>CruiseCar Manager</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:28px;background:#f7f7f7}}input,button{{font-size:15px;padding:8px;margin:4px}}table{{border-collapse:collapse;background:white;margin-top:12px}}td,th{{border:1px solid #ddd;padding:8px 12px}}pre{{background:#111;color:#0f0;padding:12px;white-space:pre-wrap}}</style></head>
<body><h2>CruiseCar Manager</h2>
<div>manager-api <input id='apiBase' style='width:360px'></div>
<div>账号 <input id='u' placeholder='username'> 密码 <input id='p' type='password' placeholder='password'><button onclick='login()'>登录/注册</button></div>
<div>Token <input id='t' style='width:520px' placeholder='login token'></div>
<h3>加入接收端</h3><input id='did' placeholder='device_id'><input id='dn' placeholder='name'><button onclick='addReceiver()'>加入</button>
<h3>接收端列表</h3><button onclick='loadReceivers()'>刷新</button><div id='devices'></div><h3>事件</h3><button onclick='loadEvents()'>刷新事件</button><pre id='log'></pre>
<script>
apiBase.value = location.protocol + '//' + location.hostname + ':{self.config.manager_port}';
async function api(path,opt={{}}){{opt.headers=Object.assign({{'Content-Type':'application/json'}},opt.headers||{{}});let tok=document.getElementById('t').value;if(tok)opt.headers.Authorization='Bearer '+tok;let r=await fetch(apiBase.value+path,opt);let j=await r.json();if(!j.ok)throw new Error(j.error||'failed');return j.data||j}}
async function login(){{let d=await api('/api/auth/login',{{method:'POST',body:JSON.stringify({{username:u.value,password:p.value}})}});t.value=d.token;loadReceivers()}}
async function addReceiver(){{await api('/api/receivers',{{method:'POST',body:JSON.stringify({{device_id:did.value,name:dn.value}})}});loadReceivers()}}
async function loadReceivers(){{let rows=await api('/api/receivers');devices.innerHTML='<table><tr><th>ID</th><th>名称</th><th>在线</th><th>ESP32</th><th>模式</th><th>地址</th></tr>'+rows.map(x=>`<tr><td>${{x.device_id}}</td><td>${{x.name||''}}</td><td>${{x.online}}</td><td>${{x.esp_connected}}</td><td>${{x.mode}}</td><td>${{x.remote_addr||''}}</td></tr>`).join('')+'</table>'}}
async function loadEvents(){{let rows=await api('/api/events?limit=50');log.textContent=JSON.stringify(rows,null,2)}}
</script></body></html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ManagerWeb().start()


if __name__ == "__main__":
    main()
