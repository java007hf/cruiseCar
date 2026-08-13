package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.cruisecar.app.data.local.ReceiverIdentityStore
import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity
import com.cruisecar.app.mvi.AppIntent
import com.cruisecar.app.mvi.MainViewModel
import org.webrtc.SurfaceViewRenderer
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

class MainActivity : Activity() {
    private val tag = "CruiseCar"
    private val appVersionLabel = "v0.1.3-camera-switch"
    private val controlPort = 42101
    private val videoPort = 42102
    private val remoteControlPort = 42110
    private val remoteWebRtcPort = 42112
    private val remoteManagerPort = 8088
    private val controlClient = ControlClient()
    private val bluetooth = BluetoothSppClient()
    private val senderExecutor = Executors.newSingleThreadExecutor()
    /* 舵机发送独立线程: 与手柄发送隔离, 手柄流量再大也不会拖慢舵机实时性 */
    private val senderServoExecutor = Executors.newSingleThreadExecutor()
    private var discoveryResponder: DiscoveryResponder? = null
    private var controlServer: ControlServer? = null
    /* 接收端转发独立线程: 把蓝牙写从 TCP 读取线程挪开, 避免 ESP32 蓝牙写阻塞
       时反向压垮发送端 TCP(导致舵机/手柄指令在发送端排队、延迟累积)。 */
    private val forwardExecutor = Executors.newSingleThreadExecutor()
    /* 接收端状态回报 + ESP32 心跳的定时任务(1Hz) */
    private val receiverReporter: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private var cameraPreview: CameraPreviewView? = null
    private var senderVideoRenderer: SurfaceViewRenderer? = null
    private var senderVideoArea: FrameLayout? = null
    private var senderLayout: LinearLayout? = null
    private var receiverVideoRenderer: SurfaceViewRenderer? = null
    private var receiverLayout: LinearLayout? = null
    private var receiverEspStatusView: TextView? = null
    /* 接收端 → 发送端 状态回报定时任务句柄, stopReceiverServices 时取消 */
    private var receiverReporterTask: ScheduledFuture<*>? = null
    private var smartFollow: SmartFollowController? = null
    private var webRtcCall: WebRtcCall? = null
    private lateinit var logView: TextView
    private var connectedReceiverHost: String? = null
    private lateinit var viewModel: MainViewModel
    private var senderMode = ControlMode.MANUAL
    private var receiverMode = ControlMode.MANUAL
    private var senderVideoEnabled = false
    private var senderVideoButton: Button? = null
    private var senderReceiverStatusView: TextView? = null
    private var senderCameraFacing = WebRtcCall.CameraFacing.FRONT
    private var senderCameraButton: Button? = null
    private var lastSenderState: GamepadState? = null
    private var lastSenderAtMs: Long = 0
    /* 舵机命令合并: 只保留"最新目标角度", 同一时刻最多排 1 个发送任务,
       避免拖动竖向舵机控件时大量回调把队列撑爆导致延迟累积。发送走独立线程,
       不与手柄发送争用同一执行器, 降低端到端延迟。 */
    private var senderServoTargetAngle = 130
    private var senderServoSendScheduled = false
    private val senderServoLock = Any()
    private var backAction: (() -> Unit)? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = MainViewModel(ReceiverIdentityStore(this))
        log("CruiseCar APK debug $appVersionLabel started")
        requestBasePermissions()
        showRoleScreen()
    }

    private fun showRoleScreen() {
        backAction = null
        val layout = rootLayout()
        layout.addView(title("CruiseCar $appVersionLabel"))
        layout.addView(button("调试台 (识别 + 遥控)") {
            startActivity(Intent(this, DebugActivity::class.java))
        })
        layout.addView(button("发送端 - 局域网") { setConnectionMode(ConnectionMode.LAN); showSenderScreen() })
        layout.addView(button("接收端 - 局域网") { setConnectionMode(ConnectionMode.LAN); showReceiverScreen() })
        layout.addView(button("发送端 - 服务器 Light") { showServerLightSetup(isSender = true) })
        layout.addView(button("接收端 - 服务器 Light") { showServerLightSetup(isSender = false) })
        layout.addView(button("发送端 - 服务器 Full") { showServerFullSetup(isSender = true) })
        layout.addView(button("接收端 - 服务器 Full") { showServerFullSetup(isSender = false) })
        setContentView(withLog(layout))
    }

    private fun showServerLightSetup(isSender: Boolean) {
        backAction = { showRoleScreen() }
        val layout = rootLayout()
        layout.addView(title("服务器 Light 配置"))
        val state = viewModel.state
        val hostInput = input("服务器 IP / 域名", state.remoteHost)
        val receiverIdentity = viewModel.receiverIdentity()
        val deviceInput = input(
            "接收端设备ID",
            state.remoteDeviceId
        )
        val senderInput = input("发送端ID", if (state.remoteSenderId.isNotBlank()) state.remoteSenderId else "phone-${System.currentTimeMillis() % 100000}")
        layout.addView(hostInput)
        if (isSender) {
            layout.addView(deviceInput)
            layout.addView(senderInput)
        } else {
            layout.addView(deviceIdentityView(receiverIdentity))
        }
        layout.addView(button(if (isSender) "进入发送端" else "确认并进入接收端") action@{
            val host = hostInput.text.toString().trim()
            val selectedDeviceId = if (isSender) deviceInput.text.toString().trim() else receiverIdentity.deviceId
            if (isSender && selectedDeviceId.isBlank()) {
                toast("请填入接收端设备ID")
                return@action
            }
            viewModel.dispatch(AppIntent.SetRemoteHost(host))
            viewModel.dispatch(AppIntent.SetRemoteDeviceId(selectedDeviceId))
            viewModel.dispatch(AppIntent.SetRemoteSenderId(senderInput.text.toString().trim()))
            viewModel.dispatch(AppIntent.SetRemoteToken(""))
            setConnectionMode(ConnectionMode.SERVER_LIGHT)
            if (isSender) showSenderScreen() else showReceiverScreen()
        })
        layout.addView(button("返回") { showRoleScreen() })
        setContentView(withLog(layout))
    }

    private fun showServerFullSetup(isSender: Boolean) {
        backAction = { showRoleScreen() }
        val layout = rootLayout()
        layout.addView(title("服务器 Full 登录"))
        val state = viewModel.state
        val hostInput = input("服务器 IP / 域名", state.remoteHost)
        val userInput = input("账号", "demo")
        val passInput = input("密码", "demo")
        val receiverIdentity = viewModel.receiverIdentity()
        layout.addView(hostInput)
        layout.addView(userInput)
        layout.addView(passInput)
        if (!isSender) {
            layout.addView(deviceIdentityView(receiverIdentity))
        }
        layout.addView(button(if (isSender) "登录并选择设备" else "登录并确认加入接收端") {
            Thread {
                try {
                    val host = hostInput.text.toString().trim()
                    val managerBaseUrl = "http://$host:$remoteManagerPort"
                    val token = RemoteApi.login(managerBaseUrl, userInput.text.toString(), passInput.text.toString())
                    viewModel.dispatch(AppIntent.SetRemoteHost(host))
                    viewModel.dispatch(AppIntent.SetRemoteManagerBaseUrl(managerBaseUrl))
                    viewModel.dispatch(AppIntent.SetRemoteToken(token))
                    setConnectionMode(ConnectionMode.SERVER_FULL)
                    if (isSender) {
                        val devices = RemoteApi.listReceivers(managerBaseUrl, token)
                        if (devices.isEmpty()) {
                            log("账号下暂无接收端，请先用接收端加入")
                        } else {
                            viewModel.dispatch(AppIntent.SetRemoteSenderId("phone-${System.currentTimeMillis() % 100000}"))
                            runOnUiThread { showDevicePicker(devices) }
                        }
                    } else {
                        viewModel.dispatch(AppIntent.SetRemoteDeviceId(receiverIdentity.deviceId))
                        RemoteApi.addReceiver(managerBaseUrl, token, receiverIdentity.deviceId, receiverIdentity.displayName)
                        log("接收端已加入账号: ${receiverIdentity.displayName} (${receiverIdentity.deviceId})")
                        runOnUiThread { showReceiverScreen() }
                    }
                } catch (e: Exception) {
                    log("Full 模式登录/配置失败: ${e.message}")
                }
            }.start()
        })
        layout.addView(button("返回") { showRoleScreen() })
        setContentView(withLog(layout))
    }

    private fun showDevicePicker(devices: List<RemoteReceiver>) {
        backAction = { showServerFullSetup(isSender = true) }
        val layout = rootLayout()
        layout.addView(title("选择接收端"))
        devices.forEach { device ->
            layout.addView(button("${device.name.ifBlank { device.deviceId }} | ${if (device.online) "在线" else "离线"} | ESP32=${device.espConnected}") {
                viewModel.dispatch(AppIntent.SetRemoteDeviceId(device.deviceId))
                log("已选择接收端: ${device.deviceId}")
                showSenderScreen()
            })
        }
        layout.addView(button("返回") { showServerFullSetup(isSender = true) })
        setContentView(withLog(layout))
    }

    private fun deviceIdentityView(identity: ReceiverIdentity): TextView = TextView(this).apply {
        text = "接收端名称：${identity.displayName}\n接收端设备ID：${identity.deviceId}\n\n接收端会自动使用该 ID 加入服务器；发送端 Full 模式登录同一账号后可直接选择设备。"
        textSize = 14f
        setTextColor(Color.rgb(70, 70, 70))
        setPadding(0, 12, 0, 12)
    }

    private fun setConnectionMode(mode: ConnectionMode) {
        viewModel.dispatch(AppIntent.SetConnectionMode(mode))
    }

    private fun showSenderScreen() {
        backAction = {
            releaseSenderCall()
            senderVideoArea = null
            senderLayout = null
            showRoleScreen()
        }
        val layout = rootLayout()
        senderLayout = layout
        layout.addView(title("发送端多模式控制"))

        val videoArea = VideoGamepadView(this)
        senderVideoArea = videoArea
        val remoteVideo = createSenderVideoRenderer(videoArea)
        videoArea.onStateChanged = stateChanged@{ state ->
            if (senderMode != ControlMode.MANUAL) {
                /* 用户在非手动模式下操作了摇杆(lx/ly 有实际输入), 自动退出智能模式回到手动遥控 */
                val controlling = state.lx != 128 || state.ly != 128 || state.buttons != 0
                if (!controlling) return@stateChanged
                val exited = senderMode
                senderMode = ControlMode.MANUAL
                sendMode(ControlMode.MANUAL)
                toast("已退出 ${exited.label} 模式，切换为手动遥控")
            }
            if (senderMode == ControlMode.MANUAL) {
                sendFromGamepad(state)
            }
        }
        videoArea.onServoChanged = { angle -> sendServoToReceiver(angle) }
        layout.addView(videoArea, LinearLayout.LayoutParams(-1, 0, 2.4f))

        /* 接收端状态(ESP32 连接态 + 接收端当前模式)由接收端 1Hz 回传, 此处展示 */
        senderReceiverStatusView = TextView(this).apply {
            text = "接收端：未连接"
            textSize = 14f
            gravity = Gravity.CENTER_HORIZONTAL
            setTextColor(Color.rgb(200, 60, 60))
        }
        layout.addView(senderReceiverStatusView)

        /* 发送端被动感知接收端状态; 接收端断开时回退 UI */
        controlClient.onStatus = { espConnected, mode ->
            runOnUiThread {
                senderReceiverStatusView?.apply {
                    text = "ESP32：${if (espConnected) "已连接" else "未连接"}  |  接收端模式：${mode.label}"
                    setTextColor(if (espConnected) Color.rgb(60, 180, 60) else Color.rgb(200, 60, 60))
                }
            }
        }
        controlClient.onReceiverGone = {
            runOnUiThread {
                senderReceiverStatusView?.apply {
                    text = "接收端：已断开"
                    setTextColor(Color.rgb(200, 60, 60))
                }
            }
        }

        layout.addView(button(senderConnectButtonText()) { connectSenderByMode() })

        layout.addView(button("远程连接 ESP32") {
            if (controlClient.isConnected()) {
                controlClient.sendCommand(ControlProtocol.CMD_CONNECT_ESP32)
                log("已向接收端下发指令：自动扫描并连接 ESP32")
            } else {
                log("未连接接收端，无法远程触发 ESP32 连接")
            }
        })

        val modeRow = row()
        modeRow.addView(button("实时视频") { startSenderVideoMode() }, weightParams())
        modeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        modeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(modeRow)
        modeRow.visibility = View.GONE
        val driveModeRow = row()
        driveModeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        driveModeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(driveModeRow)
        senderVideoButton = button(senderVideoButtonText()) { toggleSenderVideo() }
        layout.addView(senderVideoButton)
        senderCameraButton = button(senderCameraButtonText()) { toggleSenderCamera() }
        layout.addView(senderCameraButton)

        layout.addView(button("停止 / 回中") { sendFromGamepad(GamepadState(), force = true) })
        layout.addView(button("返回") {
            backAction?.invoke()
        })
        setContentView(withLog(layout))
    }

    private fun showReceiverScreen() {
        backAction = { stopReceiverServices(); receiverLayout = null; showRoleScreen() }
        val layout = rootLayout()
        receiverLayout = layout
        layout.addView(title("接收端多模式中转"))

        val preview = CameraPreviewView(this)
        cameraPreview = preview
        layout.addView(preview, LinearLayout.LayoutParams(-1, 0, 1.5f))

        receiverEspStatusView = TextView(this).apply {
            text = "未连接"
            textSize = 14f
            gravity = Gravity.CENTER_HORIZONTAL
            setTextColor(Color.rgb(200, 60, 60))
        }
        layout.addView(receiverEspStatusView)

        val blePairing = Esp32BlePairing(
            this, bluetooth,
            onLog = { log(it) },
            onToast = { msg -> Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() },
            onConnected = { updateReceiverEspStatus(true) }
        )

        layout.addView(button("自动扫描并连接 ESP32") { connectEsp32ByScan() })
        layout.addView(button("BLE 扫描附近设备") { blePairing.start() })
        layout.addView(button("启动接收服务") { startReceiverServices() })

        setContentView(withLog(layout))
        preview.start()
    }

    private fun updateReceiverEspStatus(connected: Boolean) {
        receiverEspStatusView?.apply {
            text = if (connected) "已连接" else "未连接"
            setTextColor(if (connected) Color.rgb(60, 180, 60) else Color.rgb(200, 60, 60))
        }
    }

    private fun connectEsp32ByScan() {
        Thread {
            try {
                bluetooth.connectFirstByName(this, BluetoothSppClient.ESP32_DEVICE_NAME) { log(it) }
                log("ESP32 auto-connect ready")
                runOnUiThread { updateReceiverEspStatus(true) }
            } catch (e: Exception) {
                log("ESP32 auto-connect failed: ${e.message}")
                runOnUiThread { updateReceiverEspStatus(false) }
            }
        }.start()
    }

    private fun startReceiverServices() {
        val state = viewModel.state
        if (state.connectionMode != ConnectionMode.LAN) {
            startRemoteReceiverServices()
            return
        }
        discoveryResponder = DiscoveryResponder(controlPort).also { it.start { msg -> log(msg) } }
        controlServer = ControlServer(controlPort) { packet, frame ->
            handleReceiverControlFrame(packet, frame)
        }.also { it.start { msg -> log(msg) } }

        /* 1Hz 向所有已连接发送端回传状态(ESP32 连接态 + 接收端当前模式),
           这样发送端能被动感知接收端及 ESP32 的状态, 无需轮询。 */
        receiverReporterTask = receiverReporter.scheduleAtFixedRate({
            try {
                controlServer?.send(StatusCommand.packet(bluetooth.isConnected(), receiverMode))
            } catch (e: Exception) {
                log("Status report failed: ${e.message}")
            }
        }, 0, 1, TimeUnit.SECONDS)

        log("Receiver services ready: control=$controlPort webrtc-signal=$videoPort")
    }

    private fun startRemoteReceiverServices() {
        val state = viewModel.state
        controlClient.onFrame = { packet, frame -> handleReceiverControlFrame(packet, frame) }
        Thread {
            try {
                controlClient.connectRemoteReceiver(state.remoteHost, remoteControlPort, state.remoteDeviceId, state.remoteToken) { log(it) }
                log("Remote receiver ready: device=${state.remoteDeviceId} server=${state.remoteHost}")
            } catch (e: Exception) {
                log("Remote receiver connect failed: ${e.message}")
            }
        }.start()
        receiverReporterTask = receiverReporter.scheduleAtFixedRate({
            try {
                if (controlClient.isConnected()) {
                    controlClient.sendRaw(StatusCommand.packet(bluetooth.isConnected(), receiverMode))
                }
            } catch (e: Exception) {
                log("Remote status report failed: ${e.message}")
            }
        }, 1, 1, TimeUnit.SECONDS)
    }

    private fun handleReceiverControlFrame(packet: ByteArray, frame: ControlFrame) {
        when (frame) {
            is ControlFrame.Gamepad -> {
                if (receiverMode == ControlMode.MANUAL) forwardToEsp32(packet, "Forwarded: ${packet.toHexLine()}")
            }
            is ControlFrame.Servo -> forwardToEsp32(packet, "Forwarded servo: idx=${frame.index} angle=${frame.angle}")
            is ControlFrame.Mode -> runOnUiThread {
                when (frame.mode) {
                    ControlMode.VIDEO_CALL -> setReceiverVideoEnabled(true)
                    ControlMode.VIDEO_OFF -> setReceiverVideoEnabled(false)
                    else -> applyReceiverDriveMode(frame.mode)
                }
            }
            is ControlFrame.Command -> {
                if (frame.code == ControlProtocol.CMD_CONNECT_ESP32) {
                    log("收到发送端指令：远程连接 ESP32")
                    connectEsp32ByScan()
                }
            }
            is ControlFrame.Status -> Unit
        }
    }

    private fun forwardToEsp32(packet: ByteArray, message: String) {
        if (bluetooth.isConnected()) {
            forwardExecutor.execute {
                bluetooth.send(packet)
                log(message)
            }
        } else {
            runOnUiThread { updateReceiverEspStatus(false) }
        }
    }

    private fun stopReceiverServices() {
        smartFollow?.stop()
        smartFollow = null
        receiverReporterTask?.cancel(false)
        receiverReporterTask = null
        discoveryResponder?.stop()
        controlServer?.stop()
        controlClient.onFrame = null
        discoveryResponder = null
        controlServer = null
        cameraPreview?.stop()
        cameraPreview?.visibility = View.VISIBLE
        releaseReceiverCall()
        receiverLayout = null
    }

    private fun applyReceiverDriveMode(mode: ControlMode) {
        receiverMode = mode
        smartFollow?.stop()
        smartFollow = null
        log("Receiver drive mode: ${mode.label}")

        when (mode) {
            ControlMode.MANUAL -> {
                bluetooth.send(GamepadState().toPacket())
            }
            ControlMode.SMART_FOLLOW -> {
                cameraPreview?.visibility = View.VISIBLE
                cameraPreview?.start()
                smartFollow = SmartFollowController(
                    frameProvider = { cameraPreview?.snapshot(240, 180) },
                    onState = { state ->
                        try {
                            bluetooth.send(state.toPacket())
                        } catch (e: Exception) {
                            log("Smart follow send failed: ${e.message}")
                        }
                    }
                ).also { it.start { msg -> log(msg) } }
            }
            ControlMode.PATROL -> {
                bluetooth.send(GamepadState().toPacket())
                log("Patrol mode selected; route planner is reserved for the next phase")
            }
            ControlMode.VIDEO_CALL, ControlMode.VIDEO_OFF -> Unit
        }
    }

    private fun setReceiverVideoEnabled(enabled: Boolean) {
        log("Receiver video ${if (enabled) "enabled" else "disabled"}")
        if (!enabled) {
            releaseReceiverCall()
            if (receiverMode != ControlMode.SMART_FOLLOW) {
                cameraPreview?.visibility = View.VISIBLE
                cameraPreview?.start()
            }
            return
        }
        if (webRtcCall != null) return
        if (receiverMode != ControlMode.SMART_FOLLOW) {
            cameraPreview?.stop()
            cameraPreview?.visibility = View.GONE
        }
        val renderer = createReceiverVideoRenderer()
        webRtcCall = WebRtcCall(
            this,
            WebRtcCall.Role.CALLER,
            renderer,
            cameraFacing = WebRtcCall.CameraFacing.FRONT
        ) { msg -> log(msg) }
        val state = viewModel.state
        if (state.connectionMode == ConnectionMode.LAN) {
            log("Receiver video signaling starting on $videoPort")
            webRtcCall?.startServer(videoPort)
        } else {
            log("Receiver video relay target ${state.remoteHost}:$remoteWebRtcPort room=${state.remoteDeviceId}")
            webRtcCall?.connectRelay(state.remoteHost, remoteWebRtcPort, state.remoteDeviceId, WebRtcCall.Role.CALLER, state.remoteToken)
        }
    }

    private fun senderConnectButtonText(): String = when (viewModel.state.connectionMode) {
        ConnectionMode.LAN -> "扫描接收端并连接"
        ConnectionMode.SERVER_LIGHT -> "连接服务器接收端"
        ConnectionMode.SERVER_FULL -> "连接已选设备"
    }

    private fun connectSenderByMode() {
        Thread {
            try {
                val state = viewModel.state
                if (state.connectionMode == ConnectionMode.LAN) {
                    val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                    val receivers = DiscoveryScanner(wifi).scan { log(it) }
                    if (receivers.isNotEmpty()) {
                        val receiver = receivers.first()
                        connectedReceiverHost = receiver.host
                        controlClient.connect(receiver.host, receiver.port) { log(it) }
                        sendFromGamepad(GamepadState(), force = true)
                    } else {
                        log("No receiver found")
                    }
                } else {
                    connectedReceiverHost = state.remoteHost
                    val senderId = if (state.remoteSenderId.isNotBlank()) state.remoteSenderId else "phone-${System.currentTimeMillis() % 100000}"
                    controlClient.connectRemoteSender(state.remoteHost, remoteControlPort, senderId, state.remoteDeviceId, state.remoteToken) { log(it) }
                    sendFromGamepad(GamepadState(), force = true)
                }
            } catch (e: Exception) {
                log("Connect failed: ${e.message}")
            }
        }.start()
    }

    private fun releaseReceiverCall() {
        webRtcCall?.release()
        webRtcCall = null
        receiverVideoRenderer?.let { renderer ->
            receiverLayout?.removeView(renderer)
        }
        receiverVideoRenderer = null
    }

    private fun createReceiverVideoRenderer(): SurfaceViewRenderer {
        val renderer = SurfaceViewRenderer(this).apply { visibility = View.VISIBLE }
        receiverVideoRenderer = renderer
        receiverLayout?.addView(renderer, LinearLayout.LayoutParams(-1, 0, 1.5f))
        return renderer
    }

    private fun sendMode(mode: ControlMode, remoteVideo: SurfaceViewRenderer? = null) {
        senderMode = mode
        senderExecutor.execute {
            try {
                log("sendMode ${mode.label} begin on ${threadName()}")
                if (!controlClient.isConnected()) {
                    log("Not connected: scan and connect receiver first")
                    return@execute
                }
                controlClient.sendMode(mode)
                log("Mode sent: ${mode.label}")
                if (mode != ControlMode.MANUAL) sendFromGamepad(GamepadState(), force = true)
            } catch (e: Throwable) {
                logError("Mode send failed", e)
            }
        }
    }

    private fun toggleSenderVideo() {
        if (senderVideoEnabled) {
            stopSenderVideoMode()
        } else {
            startSenderVideoMode()
        }
    }

    private fun startSenderVideoMode() {
        try {
            log("startSenderVideoMode begin on ${threadName()}")
            if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
                log("startSenderVideoMode repost to UI from ${threadName()}")
                runOnUiThread { startSenderVideoMode() }
                return
            }

            val state = viewModel.state
            val host = if (state.connectionMode == ConnectionMode.LAN) connectedReceiverHost else state.remoteHost
            if (!controlClient.isConnected() || host.isNullOrBlank()) {
                log("Not connected: scan and connect receiver first")
                return
            }

            Thread {
                try {
                    log("VIDEO_CALL mode packet send begin on ${threadName()}")
                    controlClient.sendMode(ControlMode.VIDEO_CALL)
                    log("Video command sent: ${ControlMode.VIDEO_CALL.label}")
                } catch (e: Throwable) {
                    logError("Video command send failed", e)
                }
            }.start()

            senderVideoEnabled = true
            senderVideoButton?.text = senderVideoButtonText()
            log("Sender video signaling target $host:${if (state.connectionMode == ConnectionMode.LAN) videoPort else remoteWebRtcPort}")
            log("releaseSenderCall before video on ${threadName()}")
            releaseSenderCall()
            log("createSenderVideoRenderer before video on ${threadName()}")
            val renderer = createSenderVideoRenderer()
            log("create WebRtcCall ANSWERER on ${threadName()}")
            webRtcCall = WebRtcCall(
                this,
                WebRtcCall.Role.ANSWERER,
                renderer,
                cameraFacing = senderCameraFacing
            ) { msg -> log(msg) }
            log("WebRtcCall connect call on ${threadName()}")
            if (state.connectionMode == ConnectionMode.LAN) {
                webRtcCall?.connect(host, videoPort)
            } else {
                webRtcCall?.connectRelay(host, remoteWebRtcPort, state.remoteDeviceId, WebRtcCall.Role.ANSWERER, state.remoteToken)
            }
        } catch (e: Throwable) {
            logError("startSenderVideoMode failed", e)
        }
    }

    private fun releaseSenderCall() {
        log("releaseSenderCall on ${threadName()}")
        webRtcCall?.release()
        webRtcCall = null
        senderVideoRenderer?.let { renderer ->
            (renderer.parent as? ViewGroup)?.removeView(renderer)
        }
        senderVideoRenderer = null
    }

    private fun createSenderVideoRenderer(parent: FrameLayout? = null): SurfaceViewRenderer {
        log("createSenderVideoRenderer on ${threadName()}")
        val renderer = SurfaceViewRenderer(this)
        senderVideoRenderer = renderer
        val target = parent ?: senderVideoArea
        if (target != null) {
            target.addView(renderer, 0, FrameLayout.LayoutParams(-1, -1))
        } else {
            senderLayout?.addView(renderer, LinearLayout.LayoutParams(-1, 0, 1.1f))
        }
        return renderer
    }

    private fun toggleSenderCamera() {
        senderCameraFacing = if (senderCameraFacing == WebRtcCall.CameraFacing.FRONT) {
            WebRtcCall.CameraFacing.BACK
        } else {
            WebRtcCall.CameraFacing.FRONT
        }
        senderCameraButton?.text = senderCameraButtonText()
        log("Sender camera switched to ${senderCameraFacing.name.lowercase()}")
        if (senderVideoEnabled && webRtcCall != null) {
            startSenderVideoMode()
        }
    }

    private fun stopSenderVideoMode() {
        senderVideoEnabled = false
        senderVideoButton?.text = senderVideoButtonText()
        Thread {
            try {
                controlClient.sendMode(ControlMode.VIDEO_OFF)
                log("Video command sent: ${ControlMode.VIDEO_OFF.label}")
            } catch (e: Throwable) {
                logError("Video command send failed", e)
            }
        }.start()
        releaseSenderCall()
    }

    private fun senderVideoButtonText(): String =
        if (senderVideoEnabled) "视频：开" else "视频：关"

    private fun senderCameraButtonText(): String =
        if (senderCameraFacing == WebRtcCall.CameraFacing.FRONT) {
            "发送端摄像头：前置"
        } else {
            "发送端摄像头：后置"
        }

    private fun sendFromGamepad(state: GamepadState, force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && state == lastSenderState && now - lastSenderAtMs < 80) return
        if (!force && now - lastSenderAtMs < 24) return
        lastSenderState = state
        lastSenderAtMs = now

        senderExecutor.execute {
            try {
                if (controlClient.isConnected()) {
                    controlClient.send(state)
                    log("TCP sent: ${state.toHexLine()}")
                } else {
                    log("Not connected: scan and connect receiver first")
                }
            } catch (e: Exception) {
                log("Send failed: ${e.message}")
            }
        }
    }

    private fun sendServoToReceiver(angle: Int) {
        /* 合并: 只记录最新角度; 若已有待发送的 flush 任务则跳过排程,
           等该任务发送时自然会带上最新值。这样队列里最多 1 个待发任务,
           延迟被限制在"一次 TCP 写"以内, 不会随拖动时长累积。 */
        val shouldSchedule: Boolean
        synchronized(senderServoLock) {
            senderServoTargetAngle = angle
            shouldSchedule = !senderServoSendScheduled
            senderServoSendScheduled = true
        }
        if (!shouldSchedule) return

        senderServoExecutor.execute {
            val target = synchronized(senderServoLock) {
                senderServoSendScheduled = false
                senderServoTargetAngle
            }
            try {
                if (controlClient.isConnected()) {
                    controlClient.sendServo(angle = target)
                    log("舵机指令已发送: idx=0 angle=$target")
                } else {
                    log("未连接接收端: 请先扫描并连接接收端")
                }
            } catch (e: Exception) {
                log("舵机发送失败: ${e.message}")
            }
        }
    }

    private fun requestBasePermissions() {
        val permissions = mutableListOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.CHANGE_WIFI_MULTICAST_STATE,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        )
        if (android.os.Build.VERSION.SDK_INT >= 31) {
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
        }
        val missing = permissions.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), 100)
    }

    private fun rootLayout(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(32, 32, 32, 32)
    }

    private fun row(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
    }

    private fun title(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 28f
        gravity = Gravity.CENTER_HORIZONTAL
    }

    private fun button(text: String, onClick: () -> Unit): Button = Button(this).apply {
        this.text = text
        setOnClickListener { onClick() }
    }

    private fun input(hint: String, value: String = ""): EditText = EditText(this).apply {
        this.hint = hint
        setText(value)
        setSingleLine(true)
    }

    private fun weightParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

    private fun withLog(content: LinearLayout): LinearLayout {
        logView = TextView(this).apply { textSize = 13f }
        val scroll = ScrollView(this).apply { addView(logView) }
        content.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        return content
    }

    private fun log(message: String) {
        Log.i(tag, message)
        runOnUiThread {
            if (::logView.isInitialized) {
                logView.append("$message\n")
            } else {
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }

    private fun logError(message: String, error: Throwable) {
        Log.e(tag, "$message on ${threadName()}: ${error.message}", error)
        log("$message: ${error.message}")
    }

    private fun threadName(): String =
        "${Thread.currentThread().name}/${Thread.currentThread().id}"

    override fun onBackPressed() {
        if (backAction != null) {
            backAction!!.invoke()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        stopReceiverServices()
        controlClient.close()
        bluetooth.close()
        webRtcCall?.release()
        senderExecutor.shutdownNow()
        senderServoExecutor.shutdownNow()
        forwardExecutor.shutdownNow()
        receiverReporter.shutdownNow()
        super.onDestroy()
    }
}
