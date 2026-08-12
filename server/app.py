from __future__ import annotations

import asyncio
import logging
import signal

from server.common.config import load_config
from server.common.store import Store
from server.control_server.server import ControlServer
from server.control_server.webrtc_signal import WebRtcSignalServer
from server.manager_api.server import ManagerApi


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    store = Store(config.database_path)
    control_server = ControlServer(config=config, store=store)
    webrtc_server = WebRtcSignalServer(config=config)

    loop = asyncio.get_running_loop()
    manager_api = None
    if config.deployment == "full":
        manager_api = ManagerApi(config=config, store=store, hub=control_server.hub, event_loop=loop)
        manager_api.start_in_thread()
        logging.info("deployment=full: control_server + webrtc_signal + manager-api")
    else:
        logging.info("deployment=light: control_server + webrtc_signal, manager-api disabled")

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    control_task = asyncio.create_task(control_server.start())
    webrtc_task = asyncio.create_task(webrtc_server.start())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({control_task, webrtc_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if manager_api:
        manager_api.stop()
    for task in done:
        if task in {control_task, webrtc_task}:
            task.result()


if __name__ == "__main__":
    asyncio.run(main())
