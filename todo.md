# TODO

- [x] 实现并验证 App 接收端可以通过 xiaozhi server 完成 LLM 语音对话。
- [x] 实现 MCP 控制小车，复用现有控制小车 API。

## 验证记录

- 2026-08-29：已接入本机 `xiaozhi-esp32-server-benyl` Docker 服务，Bridge 可连接 xiaozhi WebSocket，支持接收端上传 Opus 语音帧、轮询 STT/LLM/TTS/音频事件并播放下行 Opus。
- 2026-08-29：已配置 xiaozhi MCP client 调用 CruiseCar MCP Server，验证 `tools/list`、`car_move`、`car_get_status` 正常；当前接收端/ESP32 未在线时控车命令进入离线队列，符合预期。
