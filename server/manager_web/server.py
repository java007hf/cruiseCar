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
                elif raw_path == "/send":
                    self.send_response(HTTPStatus.FOUND.value)
                    self.send_header("Location", raw_path + "/")
                    self.end_headers()
                elif raw_path == "/send/":
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
      --bg: #06111f;
      --panel: rgba(8, 20, 38, .72);
      --panel-2: rgba(13, 31, 58, .78);
      --line: rgba(148, 220, 255, .18);
      --line-strong: rgba(148, 220, 255, .34);
      --text: #f3f8ff;
      --muted: #93aeca;
      --muted-2: #6f89a8;
      --blue: #38bdf8;
      --cyan: #67e8f9;
      --green: #34d399;
      --amber: #fbbf24;
      --red: #fb7185;
      --shadow: 0 26px 80px rgba(0, 0, 0, .34);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 8% 6%, rgba(56,189,248,.30), transparent 30%),
        radial-gradient(circle at 78% 0%, rgba(20,184,166,.18), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(37,99,235,.18), transparent 38%),
        linear-gradient(145deg, #06111f 0%, #0a1830 48%, #030712 100%);
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(148,220,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,220,255,.04) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, black, transparent 86%);
    }
    .shell { width: min(1240px, 100%); margin: 0 auto; padding: 26px; position: relative; }
    .hero {
      min-height: 260px;
      display: grid;
      grid-template-columns: 1.25fr .75fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .glass {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.035)), var(--panel);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(22px);
    }
    .hero-main { padding: 30px; position: relative; overflow: hidden; }
    .hero-main::after {
      content: "";
      position: absolute;
      right: -90px;
      top: -80px;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(103,232,249,.35), transparent 68%);
    }
    .brand-row { display: flex; align-items: center; gap: 14px; margin-bottom: 24px; position: relative; z-index: 1; }
    .logo {
      width: 56px; height: 56px; display: grid; place-items: center; border-radius: 20px;
      background: linear-gradient(135deg, rgba(56,189,248,.32), rgba(103,232,249,.12));
      border: 1px solid var(--line-strong); font-size: 25px;
    }
    .eyebrow { color: var(--cyan); font-size: 13px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(36px, 5vw, 64px); line-height: .98; letter-spacing: -2px; position: relative; z-index: 1; }
    .subtitle { max-width: 680px; margin: 18px 0 26px; color: #b8c9dd; font-size: 16px; line-height: 1.7; position: relative; z-index: 1; }
    .hero-actions, .actions { display: flex; flex-wrap: wrap; gap: 10px; position: relative; z-index: 1; }
    button, .link-btn {
      border: 0; border-radius: 999px; padding: 12px 17px; min-height: 46px;
      color: #04111f; background: linear-gradient(135deg, #e0f7ff, #67e8f9 54%, #38bdf8);
      font-weight: 900; font-size: 14px; cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      box-shadow: 0 14px 34px rgba(56,189,248,.22);
      transition: transform .16s ease, border-color .16s ease, background .16s ease;
    }
    button:hover, .link-btn:hover { transform: translateY(-1px); }
    button.secondary, .link-btn.secondary {
      color: var(--text); background: rgba(255,255,255,.08); border: 1px solid var(--line-strong); box-shadow: none;
    }
    .hero-side { padding: 22px; display: grid; gap: 12px; }
    .signal-card { padding: 18px; border-radius: 22px; background: rgba(255,255,255,.055); border: 1px solid var(--line); }
    .signal-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 13px; margin-bottom: 10px; }
    .signal-value { font-size: 22px; font-weight: 950; letter-spacing: -.5px; word-break: break-all; }
    .pulse { width: 11px; height: 11px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 8px rgba(52,211,153,.12); }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
    .metric { padding: 18px; min-height: 118px; display: flex; flex-direction: column; justify-content: space-between; }
    .metric .name { color: var(--muted); font-weight: 800; font-size: 13px; }
    .metric .num { font-size: 34px; font-weight: 950; letter-spacing: -1px; }
    .metric .hint { color: var(--muted-2); font-size: 12px; }
    .layout { display: grid; grid-template-columns: 350px 1fr; gap: 18px; align-items: start; }
    .stack { display: grid; gap: 14px; }
    .card { padding: 20px; }
    .card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
    .card h2 { margin: 0; font-size: 18px; letter-spacing: -.2px; }
    .card-note { color: var(--muted-2); font-size: 12px; margin-top: 4px; }
    label { display: block; margin: 12px 0 7px; color: var(--muted); font-size: 12px; font-weight: 900; letter-spacing: .02em; }
    input {
      width: 100%; border: 1px solid var(--line); border-radius: 16px; padding: 13px 14px;
      background: rgba(2, 8, 23, .45); color: var(--text); outline: none; font-size: 15px;
      transition: border-color .16s ease, background .16s ease;
    }
    input:focus { border-color: rgba(103,232,249,.72); background: rgba(2,8,23,.62); }
    input::placeholder { color: rgba(147,174,202,.55); }
    .actions > * { flex: 1; min-width: 120px; }
    .device-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .device {
      position: relative; overflow: hidden; min-height: 168px; display: flex; flex-direction: column; justify-content: space-between;
      padding: 18px; border-radius: 24px; background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035)); border: 1px solid var(--line);
    }
    .device::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: linear-gradient(90deg, var(--blue), transparent); opacity: .75; }
    .device-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .device-title { font-weight: 950; font-size: 17px; line-height: 1.25; word-break: break-all; }
    .device-sub { color: var(--muted); margin-top: 8px; font-size: 12px; line-height: 1.45; word-break: break-all; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .badge { padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 900; color: #c7d8ea; background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.08); }
    .badge.ok { color: #032014; background: var(--green); border-color: transparent; }
    .badge.warn { color: #2b1900; background: var(--amber); border-color: transparent; }
    .badge.danger { color: #fff; background: rgba(251,113,133,.22); border-color: rgba(251,113,133,.38); }
    .device button { min-height: 38px; padding: 9px 12px; font-size: 12px; }
    .empty { padding: 34px; text-align: center; color: var(--muted); border: 1px dashed var(--line-strong); border-radius: 24px; background: rgba(255,255,255,.035); grid-column: 1 / -1; }
    pre {
      margin: 0; min-height: 300px; max-height: 460px; overflow: auto; white-space: pre-wrap;
      color: #cfe7ff; background: rgba(2, 8, 23, .55); border: 1px solid var(--line); border-radius: 22px;
      padding: 16px; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .toast {
      position: fixed; right: 22px; bottom: 22px; max-width: min(420px, calc(100vw - 32px)); padding: 14px 16px; border-radius: 18px;
      background: rgba(3, 10, 25, .94); border: 1px solid var(--line-strong); color: var(--text); box-shadow: var(--shadow);
      opacity: 0; transform: translateY(10px); transition: .22s ease; z-index: 10;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    @media (max-width: 980px) { .hero, .layout { grid-template-columns: 1fr; } .device-list { grid-template-columns: 1fr; } }
    @media (max-width: 680px) { .shell { padding: 14px; } .hero-main { padding: 22px; } .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } h1 { font-size: 38px; } }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-main glass">
        <div class="brand-row">
          <div class="logo">◈</div>
          <div>
            <div class="eyebrow">CruiseCar Console</div>
            <div class="card-note">远程控制服务 / 设备在线状态 / Web 发送端</div>
          </div>
        </div>
        <h1>车辆设备<br />远程管理台</h1>
        <p class="subtitle">统一管理同账号下的接收端设备，查看在线与 ESP32 连接状态，快速进入手机网页发送端进行遥控和 WebRTC 视频查看。</p>
        <div class="hero-actions">
          <a class="link-btn" href="/send/" target="_blank">打开 Web 发送端</a>
          <button class="secondary" onclick="loadReceivers()">刷新设备</button>
          <button class="secondary" onclick="loadEvents()">刷新事件</button>
        </div>
      </div>
      <aside class="hero-side glass">
        <div class="signal-card">
          <div class="signal-title"><span>Manager API</span><span class="pulse"></span></div>
          <div id="apiPreview" class="signal-value">--</div>
        </div>
        <div class="signal-card">
          <div class="signal-title"><span>当前账号</span><span>USER</span></div>
          <div id="accountPreview" class="signal-value">未登录</div>
        </div>
        <div class="signal-card">
          <div class="signal-title"><span>快速入口</span><span>WEB</span></div>
          <div class="actions">
            <a class="link-btn secondary" href="/send/" target="_blank">/send/</a>
          </div>
        </div>
      </aside>
    </section>

    <section class="metrics">
      <div class="metric glass"><div class="name">接收端总数</div><div id="totalCount" class="num">0</div><div class="hint">账号下注册设备</div></div>
      <div class="metric glass"><div class="name">在线设备</div><div id="onlineCount" class="num">0</div><div class="hint">已连控制通道</div></div>
      <div class="metric glass"><div class="name">ESP32 已连</div><div id="espCount" class="num">0</div><div class="hint">蓝牙链路可用</div></div>
      <div class="metric glass"><div class="name">最近事件</div><div id="eventCount" class="num">0</div><div class="hint">最新 50 条</div></div>
    </section>

    <section class="layout">
      <aside class="stack">
        <div class="card glass">
          <div class="card-head">
            <div>
              <h2>账号连接</h2>
              <div class="card-note">登录后自动保存 token 到浏览器本地</div>
            </div>
          </div>
          <label for="apiBase">API 地址</label>
          <input id="apiBase" />
          <label for="u">账号</label>
          <input id="u" placeholder="username" autocomplete="username" />
          <label for="p">密码</label>
          <input id="p" type="password" placeholder="password" autocomplete="current-password" />
          <label for="t">Token</label>
          <input id="t" placeholder="登录后自动填入，也可手动粘贴" />
          <div class="actions" style="margin-top:16px">
            <button onclick="login()">登录 / 注册</button>
            <button class="secondary" onclick="loadReceivers()">刷新</button>
          </div>
        </div>

        <div class="card glass">
          <div class="card-head">
            <div>
              <h2>手动加入接收端</h2>
              <div class="card-note">通常 App 接收端会自动注册，这里用于调试</div>
            </div>
          </div>
          <label for="did">Device ID</label>
          <input id="did" placeholder="car-phone-xxx" />
          <label for="dn">设备名称</label>
          <input id="dn" placeholder="客厅小车 / 手机接收端" />
          <div class="actions" style="margin-top:16px">
            <button onclick="addReceiver()">加入账号</button>
          </div>
        </div>
      </aside>

      <section class="stack">
        <div class="card glass">
          <div class="card-head">
            <div>
              <h2>接收端设备</h2>
              <div class="card-note">查看在线状态、ESP32 连接状态和当前模式</div>
            </div>
            <button class="secondary" onclick="loadReceivers()">刷新列表</button>
          </div>
          <div id="devices" class="device-list"><div class="empty">登录后查看账号下的接收端设备</div></div>
        </div>

        <div class="card glass">
          <div class="card-head">
            <div>
              <h2>事件日志</h2>
              <div class="card-note">展示最近控制指令、设备上下线和状态变化</div>
            </div>
            <button class="secondary" onclick="loadEvents()">刷新事件</button>
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
    updatePreview();

    function persist() {
      localStorage.setItem(storageKey, JSON.stringify({ apiBase: apiBase.value, username: u.value, token: t.value }));
      updatePreview();
    }

    apiBase.addEventListener('input', updatePreview);
    u.addEventListener('input', updatePreview);
    t.addEventListener('input', updatePreview);

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

    async function deleteReceiver(deviceId, name, online) {
      const label = name || deviceId;
      if (online) {
        notify('在线接收端需要先退出接收端页面后再删除');
        return;
      }
      if (!confirm('确定删除接收端「' + label + '」？')) return;
      try {
        await api('/api/receivers/' + encodeURIComponent(deviceId), { method: 'DELETE' });
        notify('接收端已删除');
        await loadReceivers();
        await loadEvents();
      } catch (error) { notify('删除失败：' + error.message); }
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
      const online = x.online ? '<span class="badge ok">在线</span>' : '<span class="badge danger">离线</span>';
      const esp = x.esp_connected ? '<span class="badge ok">ESP32 已连接</span>' : '<span class="badge warn">ESP32 未连接</span>';
      return `<article class="device">
        <div class="device-top">
          <div>
            <div class="device-title">${escapeHtml(x.name || x.device_id)}</div>
            <div class="device-sub">${escapeHtml(x.device_id)}${x.remote_addr ? '<br />' + escapeHtml(x.remote_addr) : ''}</div>
          </div>
          <div class="actions" style="flex:0 0 auto; gap:8px">
            <button class="secondary" onclick="copyText('${escapeAttr(x.device_id)}')">复制</button>
            <button class="secondary" onclick="deleteReceiver('${escapeAttr(x.device_id)}','${escapeAttr(x.name || x.device_id)}',${x.online ? 'true' : 'false'})">删除</button>
          </div>
        </div>
        <div class="badges">${online}${esp}<span class="badge">${escapeHtml(x.mode || 'manual')}</span></div>
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

    function updatePreview() {
      apiPreview.textContent = apiBase.value || '--';
      accountPreview.textContent = u.value ? (t.value ? u.value + ' · 已授权' : u.value + ' · 未登录') : '未登录';
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
