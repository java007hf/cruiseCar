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
                raw_path = parsed.path or "/"
                path = raw_path.rstrip("/") or "/"
                if path == "/health":
                    self._send_text("ok\n", "text/plain; charset=utf-8")
                elif path == "/":
                    self._send_text(web.index_html(), "text/html; charset=utf-8")
                elif raw_path in {"/send", "/web_send"}:
                    self.send_response(HTTPStatus.FOUND.value)
                    self.send_header("Location", raw_path + "/")
                    self.end_headers()
                elif raw_path in {"/send/", "/web_send/"}:
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
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CruiseCar Manager</title>
  <style>
    :root {
      color-scheme: dark;
      --bg0: #07111f;
      --bg1: #0b2447;
      --bg2: #031525;
      --card: rgba(255, 255, 255, 0.09);
      --card-strong: rgba(255, 255, 255, 0.14);
      --line: rgba(186, 230, 253, 0.28);
      --text: #f8fbff;
      --muted: #bae6fd;
      --soft: #7dd3fc;
      --accent: #e0f2fe;
      --accent-text: #082f49;
      --green: #22c55e;
      --yellow: #fbbf24;
      --red: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 14% 12%, rgba(56, 189, 248, .24), transparent 34%),
        radial-gradient(circle at 90% 20%, rgba(99, 102, 241, .22), transparent 34%),
        linear-gradient(135deg, var(--bg0), var(--bg1) 52%, var(--bg2));
    }
    .shell { width: min(1180px, 100%); margin: 0 auto; padding: 28px; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 24px; }
    .brand { display: flex; align-items: center; gap: 14px; }
    .logo { width: 52px; height: 52px; display: grid; place-items: center; border-radius: 18px; background: var(--card-strong); border: 1px solid var(--line); box-shadow: 0 18px 48px rgba(0,0,0,.22); }
    h1 { margin: 0; font-size: 30px; letter-spacing: -.5px; }
    .subtitle { margin-top: 4px; color: var(--muted); font-size: 14px; }
    .top-actions { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .glass { background: var(--card); border: 1px solid var(--line); border-radius: 26px; backdrop-filter: blur(18px); box-shadow: 0 18px 48px rgba(0,0,0,.18); }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 18px; align-items: start; }
    .card { padding: 20px; }
    .card h2 { margin: 0 0 14px; font-size: 18px; }
    .stack { display: grid; gap: 12px; }
    label { display: block; margin: 2px 0 6px; color: var(--muted); font-size: 13px; font-weight: 700; }
    input {
      width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 12px 13px;
      background: rgba(255, 255, 255, .08); color: var(--text); outline: none; font-size: 15px;
    }
    input::placeholder { color: rgba(224, 242, 254, .45); }
    button, .link-btn {
      border: 0; border-radius: 999px; padding: 12px 16px; color: var(--accent-text); background: var(--accent);
      font-weight: 800; font-size: 14px; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    }
    button.secondary, .link-btn.secondary { color: var(--text); background: rgba(255,255,255,.12); border: 1px solid var(--line); }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; }
    .actions > * { flex: 1; min-width: 120px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .metric { padding: 16px; }
    .metric .num { font-size: 28px; font-weight: 900; letter-spacing: -.6px; }
    .metric .name { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    .device-list { display: grid; gap: 12px; }
    .device { display: grid; grid-template-columns: 1fr auto; gap: 14px; padding: 16px; border-radius: 20px; background: rgba(255,255,255,.08); border: 1px solid rgba(186,230,253,.20); }
    .device-title { font-weight: 900; font-size: 16px; word-break: break-all; }
    .device-sub { color: var(--muted); margin-top: 6px; font-size: 13px; word-break: break-all; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .badge { padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; color: var(--muted); background: rgba(255,255,255,.10); }
    .badge.ok { color: #052e16; background: var(--green); }
    .badge.warn { color: #422006; background: var(--yellow); }
    .empty { padding: 26px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 20px; }
    pre { margin: 0; min-height: 280px; max-height: 420px; overflow: auto; white-space: pre-wrap; color: #dbeafe; background: rgba(0,0,0,.22); border-radius: 18px; padding: 16px; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .toast { position: fixed; right: 22px; bottom: 22px; max-width: 420px; padding: 14px 16px; border-radius: 16px; background: rgba(2, 6, 23, .92); border: 1px solid var(--line); color: var(--text); box-shadow: 0 18px 50px rgba(0,0,0,.32); opacity: 0; transform: translateY(10px); transition: .22s ease; }
    .toast.show { opacity: 1; transform: translateY(0); }
    @media (max-width: 860px) { .shell { padding: 16px; } .topbar { align-items: flex-start; flex-direction: column; } .grid { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div class="brand">
        <div class="logo">✦</div>
        <div>
          <h1>CruiseCar Manager</h1>
          <div class="subtitle">账号设备管理 · 接收端监控 · 网页发送端入口</div>
        </div>
      </div>
      <div class="top-actions">
        <a class="link-btn secondary" href="/send/" target="_blank">打开 Web 发送端</a>
        <button class="secondary" onclick="loadEvents()">刷新事件</button>
      </div>
    </section>

    <section class="metrics">
      <div class="metric glass"><div id="totalCount" class="num">0</div><div class="name">接收端</div></div>
      <div class="metric glass"><div id="onlineCount" class="num">0</div><div class="name">在线设备</div></div>
      <div class="metric glass"><div id="espCount" class="num">0</div><div class="name">ESP32 已连</div></div>
      <div class="metric glass"><div id="eventCount" class="num">0</div><div class="name">最近事件</div></div>
    </section>

    <section class="grid">
      <aside class="stack">
        <div class="card glass">
          <h2>连接 Manager API</h2>
          <label for="apiBase">API 地址</label>
          <input id="apiBase" />
          <label for="u">账号</label>
          <input id="u" placeholder="username" autocomplete="username" />
          <label for="p">密码</label>
          <input id="p" type="password" placeholder="password" autocomplete="current-password" />
          <label for="t">Token</label>
          <input id="t" placeholder="登录后自动填入，也可手动粘贴" />
          <div class="actions" style="margin-top:14px">
            <button onclick="login()">登录 / 注册</button>
            <button class="secondary" onclick="loadReceivers()">刷新设备</button>
          </div>
        </div>

        <div class="card glass">
          <h2>手动加入接收端</h2>
          <label for="did">Device ID</label>
          <input id="did" placeholder="car-phone-xxx" />
          <label for="dn">设备名称</label>
          <input id="dn" placeholder="客厅小车 / 手机接收端" />
          <div class="actions" style="margin-top:14px">
            <button onclick="addReceiver()">加入账号</button>
          </div>
        </div>
      </aside>

      <section class="stack">
        <div class="card glass">
          <div class="toolbar">
            <h2 style="margin:0">接收端列表</h2>
            <button class="secondary" onclick="loadReceivers()">刷新</button>
          </div>
          <div id="devices" class="device-list"><div class="empty">登录后查看账号下的接收端设备</div></div>
        </div>

        <div class="card glass">
          <div class="toolbar">
            <h2 style="margin:0">最近事件</h2>
            <button class="secondary" onclick="loadEvents()">刷新</button>
          </div>
          <pre id="log">暂无事件</pre>
        </div>
      </section>
    </section>
  </main>
  <div id="toast" class="toast"></div>

  <script>
    const storageKey = 'cruisecar.manager.v2';
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
    apiBase.value = saved.apiBase || (location.protocol + '//' + location.hostname + ':__MANAGER_PORT__');
    u.value = saved.username || '';
    t.value = saved.token || '';

    function persist() {
      localStorage.setItem(storageKey, JSON.stringify({ apiBase: apiBase.value, username: u.value, token: t.value }));
    }

    function notify(message) {
      toast.textContent = message;
      toast.classList.add('show');
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
    }

    async function api(path, opt = {}) {
      persist();
      opt.headers = Object.assign({'Content-Type': 'application/json'}, opt.headers || {});
      const tok = document.getElementById('t').value.trim();
      if (tok) opt.headers.Authorization = 'Bearer ' + tok;
      const response = await fetch(apiBase.value.replace(new RegExp('/+$'), '') + path, opt);
      const json = await response.json().catch(() => ({}));
      if (!response.ok || !json.ok) throw new Error(json.error || '请求失败');
      return json.data || json;
    }

    async function login() {
      try {
        const data = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ username: u.value, password: p.value }) });
        t.value = data.token;
        p.value = '';
        persist();
        notify('登录成功');
        await loadReceivers();
        await loadEvents();
      } catch (error) { notify('登录失败：' + error.message); }
    }

    async function addReceiver() {
      try {
        await api('/api/receivers', { method: 'POST', body: JSON.stringify({ device_id: did.value, name: dn.value }) });
        notify('接收端已加入账号');
        await loadReceivers();
      } catch (error) { notify('加入失败：' + error.message); }
    }

    async function loadReceivers() {
      try {
        const rows = await api('/api/receivers');
        totalCount.textContent = rows.length;
        onlineCount.textContent = rows.filter(x => x.online).length;
        espCount.textContent = rows.filter(x => x.esp_connected).length;
        if (!rows.length) {
          devices.innerHTML = '<div class="empty">暂无接收端，请先在 App 接收端登录并加入账号</div>';
          return;
        }
        devices.innerHTML = rows.map(deviceCard).join('');
        notify('设备列表已刷新');
      } catch (error) { notify('刷新设备失败：' + error.message); }
    }

    function deviceCard(x) {
      const online = x.online ? '<span class="badge ok">在线</span>' : '<span class="badge">离线</span>';
      const esp = x.esp_connected ? '<span class="badge ok">ESP32 已连接</span>' : '<span class="badge warn">ESP32 未连接</span>';
      return `<article class="device">
        <div>
          <div class="device-title">${escapeHtml(x.name || x.device_id)}</div>
          <div class="device-sub">${escapeHtml(x.device_id)}${x.remote_addr ? ' · ' + escapeHtml(x.remote_addr) : ''}</div>
          <div class="badges">${online}${esp}<span class="badge">${escapeHtml(x.mode || 'manual')}</span></div>
        </div>
        <button class="secondary" onclick="copyText('${escapeAttr(x.device_id)}')">复制 ID</button>
      </article>`;
    }

    async function loadEvents() {
      try {
        const rows = await api('/api/events?limit=50');
        eventCount.textContent = rows.length;
        log.textContent = JSON.stringify(rows, null, 2);
      } catch (error) { notify('刷新事件失败：' + error.message); }
    }

    function copyText(text) {
      navigator.clipboard?.writeText(text);
      notify('已复制设备 ID');
    }

    function escapeHtml(value) {
      return String(value || '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    }
    function escapeAttr(value) { return escapeHtml(value).replace(/`/g, '&#96;'); }
    if (t.value) { loadReceivers(); loadEvents(); }
  </script>
</body>
</html>""".replace("__MANAGER_PORT__", str(self.config.manager_port))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ManagerWeb().start()


if __name__ == "__main__":
    main()
