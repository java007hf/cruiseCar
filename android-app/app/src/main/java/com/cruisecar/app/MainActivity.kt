package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import android.os.Bundle
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
import org.webrtc.SurfaceViewRenderer
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val tag = "CruiseCar"
    private val appVersionLabel = "v0.1.3-camera-switch"
    private val controlPort = 42101
    private val videoPort = 42102
    private val controlClient = ControlClient()
    private val bluetooth = BluetoothSppClient()
    private val senderExecutor = Executors.newSingleThreadExecutor()
    private var discoveryResponder: DiscoveryResponder? = null
    private var controlServer: ControlServer? = null
    private var cameraPreview: CameraPreviewView? = null
    private var senderVideoRenderer: SurfaceViewRenderer? = null
    private var senderVideoArea: FrameLayout? = null
    private var senderLayout: LinearLayout? = null
    private var receiverVideoRenderer: SurfaceViewRenderer? = null
    private var receiverLayout: LinearLayout? = null
    private var smartFollow: SmartFollowController? = null
    private var webRtcCall: WebRtcCall? = null
    private lateinit var logView: TextView
    private var connectedReceiverHost: String? = null
    private var senderMode = ControlMode.MANUAL
    private var receiverMode = ControlMode.MANUAL
    private var senderVideoEnabled = false
    private var senderVideoButton: Button? = null
    private var senderCameraFacing = WebRtcCall.CameraFacing.FRONT
    private var senderCameraButton: Button? = null
    private var lastSenderState: GamepadState? = null
    private var lastSenderAtMs: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        log("CruiseCar APK debug $appVersionLabel started")
        requestBasePermissions()
        showRoleScreen()
    }

    private fun showRoleScreen() {
        val layout = rootLayout()
        layout.addView(title("CruiseCar $appVersionLabel"))
        layout.addView(button("发送端") { showSenderScreen() })
        layout.addView(button("接收端") { showReceiverScreen() })
        setContentView(withLog(layout))
    }

    private fun showSenderScreen() {
        val layout = rootLayout()
        senderLayout = layout
        layout.addView(title("发送端多模式控制"))

        val videoArea = FrameLayout(this)
        senderVideoArea = videoArea
        val remoteVideo = createSenderVideoRenderer(videoArea)
        val gamepad = GamepadView(this).apply {
            onStateChanged = { state ->
                if (senderMode == ControlMode.MANUAL) {
                    sendFromGamepad(state)
                }
            }
        }
        videoArea.addView(gamepad, FrameLayout.LayoutParams(-1, -1))
        layout.addView(videoArea, LinearLayout.LayoutParams(-1, 0, 2.4f))

        layout.addView(button("扫描接收端并连接") {
            Thread {
                try {
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
                } catch (e: Exception) {
                    log("Scan/connect failed: ${e.message}")
                }
            }.start()
        })

        val modeRow = row()
        modeRow.addView(button("实时视频") { startSenderVideoMode() }, weightParams())
        modeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        modeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(modeRow)
        modeRow.visibility = View.GONE
        val driveModeRow = row()
        driveModeRow.addView(button("手动遥控") { sendMode(ControlMode.MANUAL, remoteVideo) }, weightParams())
        driveModeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        driveModeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(driveModeRow)
        senderVideoButton = button(senderVideoButtonText()) { toggleSenderVideo() }
        layout.addView(senderVideoButton)
        senderCameraButton = button(senderCameraButtonText()) { toggleSenderCamera() }
        layout.addView(senderCameraButton)
        layout.addView(button("停止 / 回中") { sendFromGamepad(GamepadState(), force = true) })
        layout.addView(button("返回") {
            releaseSenderCall()
            senderVideoArea = null
            senderLayout = null
            showRoleScreen()
        })
        setContentView(withLog(layout))
    }

    private fun showReceiverScreen() {
        val layout = rootLayout()
        receiverLayout = layout
        layout.addView(title("接收端多模式中转"))

        val preview = CameraPreviewView(this)
        cameraPreview = preview
        layout.addView(preview, LinearLayout.LayoutParams(-1, 0, 1.5f))

        val addressInput = EditText(this)
        addressInput.hint = "ESP32 蓝牙地址，例如 AA:BB:CC:DD:EE:FF"
        layout.addView(addressInput)
        layout.addView(button("自动扫描并连接 ESP32") {
            connectEsp32ByScan()
        })
        layout.addView(button("连接 ESP32 SPP") {
            Thread {
                try {
                    bluetooth.connect(addressInput.text.toString().trim()) { log(it) }
                } catch (e: Exception) {
                    log("Bluetooth connect failed: ${e.message}")
                }
            }.start()
        })
        layout.addView(button("启动接收服务") { startReceiverServices() })

        setContentView(withLog(layout))
        preview.start()
    }

    private fun connectEsp32ByScan() {
        Thread {
            try {
                bluetooth.connectFirstByName(this, BluetoothSppClient.ESP32_DEVICE_NAME) { log(it) }
                log("ESP32 auto-connect ready")
            } catch (e: Exception) {
                log("ESP32 auto-connect failed: ${e.message}")
            }
        }.start()
    }

    private fun startReceiverServices() {
        discoveryResponder = DiscoveryResponder(controlPort).also { it.start { msg -> log(msg) } }
        controlServer = ControlServer(controlPort) { packet, frame ->
            when (frame) {
                is ControlFrame.Gamepad -> {
                    if (receiverMode == ControlMode.MANUAL) {
                        bluetooth.send(packet)
                        log("Forwarded: ${packet.toHexLine()}")
                    }
                }
                is ControlFrame.Mode -> runOnUiThread {
                    when (frame.mode) {
                        ControlMode.VIDEO_CALL -> setReceiverVideoEnabled(true)
                        ControlMode.VIDEO_OFF -> setReceiverVideoEnabled(false)
                        else -> applyReceiverDriveMode(frame.mode)
                    }
                }
            }
        }.also { it.start { msg -> log(msg) } }
        log("Receiver services ready: control=$controlPort webrtc-signal=$videoPort")
    }

    private fun stopReceiverServices() {
        smartFollow?.stop()
        smartFollow = null
        discoveryResponder?.stop()
        controlServer?.stop()
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
        log("Receiver video signaling starting on $videoPort")
        webRtcCall?.startServer(videoPort)
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

            val host = connectedReceiverHost
            if (!controlClient.isConnected() || host == null) {
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
            log("Sender video signaling target $host:$videoPort")
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
            webRtcCall?.connect(host, videoPort)
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

    private fun logError(message: String, error: Throwable) {
        Log.e(tag, "$message on ${threadName()}: ${error.message}", error)
        log("$message: ${error.message}")
    }

    private fun threadName(): String =
        "${Thread.currentThread().name}/${Thread.currentThread().id}"

    override fun onDestroy() {
        stopReceiverServices()
        controlClient.close()
        bluetooth.close()
        webRtcCall?.release()
        senderExecutor.shutdownNow()
        super.onDestroy()
    }
}
