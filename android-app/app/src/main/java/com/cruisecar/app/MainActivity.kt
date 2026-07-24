package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.webrtc.SurfaceViewRenderer
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val controlPort = 42101
    private val videoPort = 42102
    private val controlClient = ControlClient()
    private val bluetooth = BluetoothSppClient()
    private val senderExecutor = Executors.newSingleThreadExecutor()
    private var discoveryResponder: DiscoveryResponder? = null
    private var controlServer: ControlServer? = null
    private var cameraPreview: CameraPreviewView? = null
    private var smartFollow: SmartFollowController? = null
    private var webRtcCall: WebRtcCall? = null
    private lateinit var logView: TextView
    private var connectedReceiverHost: String? = null
    private var senderMode = ControlMode.MANUAL
    private var receiverMode = ControlMode.MANUAL
    private var lastSenderState: GamepadState? = null
    private var lastSenderAtMs: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestBasePermissions()
        showRoleScreen()
    }

    private fun showRoleScreen() {
        val layout = rootLayout()
        layout.addView(title("CruiseCar"))
        layout.addView(button("发送端") { showSenderScreen() })
        layout.addView(button("接收端") { showReceiverScreen() })
        setContentView(withLog(layout))
    }

    private fun showSenderScreen() {
        val layout = rootLayout()
        layout.addView(title("发送端多模式控制"))

        val remoteVideo = SurfaceViewRenderer(this)
        layout.addView(remoteVideo, LinearLayout.LayoutParams(-1, 0, 1.1f))

        layout.addView(button("扫描接收端并连接") {
            Thread {
                try {
                    val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                    val receivers = DiscoveryScanner(wifi).scan { log(it) }
                    if (receivers.isNotEmpty()) {
                        val receiver = receivers.first()
                        connectedReceiverHost = receiver.host
                        controlClient.connect(receiver.host, receiver.port) { log(it) }
                        sendMode(ControlMode.MANUAL, remoteVideo)
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
        modeRow.addView(button("手动遥控") { sendMode(ControlMode.MANUAL, remoteVideo) }, weightParams())
        modeRow.addView(button("实时视频") { sendMode(ControlMode.VIDEO_CALL, remoteVideo) }, weightParams())
        modeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        modeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(modeRow)

        val gamepad = GamepadView(this).apply {
            onStateChanged = { state ->
                if (senderMode == ControlMode.MANUAL) {
                    sendFromGamepad(state)
                }
            }
        }
        layout.addView(gamepad, LinearLayout.LayoutParams(-1, 0, 1.7f))
        layout.addView(button("停止 / 回中") { sendFromGamepad(GamepadState(), force = true) })
        layout.addView(button("返回") {
            webRtcCall?.release()
            webRtcCall = null
            showRoleScreen()
        })
        setContentView(withLog(layout))
    }

    private fun showReceiverScreen() {
        val layout = rootLayout()
        layout.addView(title("接收端多模式中转"))

        val preview = CameraPreviewView(this)
        cameraPreview = preview
        layout.addView(preview, LinearLayout.LayoutParams(-1, 0, 1.5f))

        val addressInput = EditText(this)
        addressInput.hint = "ESP32 蓝牙地址，例如 AA:BB:CC:DD:EE:FF"
        layout.addView(addressInput)
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

        val localModeRow = row()
        localModeRow.addView(button("手动") { applyReceiverMode(ControlMode.MANUAL) }, weightParams())
        localModeRow.addView(button("视频") { applyReceiverMode(ControlMode.VIDEO_CALL) }, weightParams())
        localModeRow.addView(button("跟随") { applyReceiverMode(ControlMode.SMART_FOLLOW) }, weightParams())
        localModeRow.addView(button("巡逻") { applyReceiverMode(ControlMode.PATROL) }, weightParams())
        layout.addView(localModeRow)

        layout.addView(button("停止") {
            stopReceiverServices()
            bluetooth.close()
            log("Receiver stopped")
        })
        layout.addView(button("返回") {
            stopReceiverServices()
            showRoleScreen()
        })
        setContentView(withLog(layout))
        preview.start()
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
                is ControlFrame.Mode -> applyReceiverMode(frame.mode)
            }
        }.also { it.start { msg -> log(msg) } }
        log("Receiver services ready: control=$controlPort webrtc-signal=$videoPort")
    }

    private fun stopReceiverServices() {
        smartFollow?.stop()
        smartFollow = null
        webRtcCall?.release()
        webRtcCall = null
        discoveryResponder?.stop()
        controlServer?.stop()
        discoveryResponder = null
        controlServer = null
        cameraPreview?.stop()
    }

    private fun applyReceiverMode(mode: ControlMode) {
        receiverMode = mode
        smartFollow?.stop()
        smartFollow = null
        log("Receiver mode: ${mode.label}")

        when (mode) {
            ControlMode.MANUAL -> {
                webRtcCall?.release()
                webRtcCall = null
                bluetooth.send(GamepadState().toPacket())
            }
            ControlMode.VIDEO_CALL -> {
                cameraPreview?.stop()
                webRtcCall?.release()
                val renderer = SurfaceViewRenderer(this)
                webRtcCall = WebRtcCall(this, WebRtcCall.Role.CALLER, renderer) { msg -> log(msg) }
                webRtcCall?.startServer(videoPort)
            }
            ControlMode.SMART_FOLLOW -> {
                webRtcCall?.release()
                webRtcCall = null
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
                webRtcCall?.release()
                webRtcCall = null
                bluetooth.send(GamepadState().toPacket())
                log("Patrol mode selected; route planner is reserved for the next phase")
            }
        }
    }

    private fun sendMode(mode: ControlMode, remoteVideo: SurfaceViewRenderer? = null) {
        senderMode = mode
        senderExecutor.execute {
            try {
                if (!controlClient.isConnected()) {
                    log("Not connected: scan and connect receiver first")
                    return@execute
                }
                controlClient.sendMode(mode)
                log("Mode sent: ${mode.label}")

                val host = connectedReceiverHost
                if (mode == ControlMode.VIDEO_CALL && host != null && remoteVideo != null) {
                    webRtcCall?.release()
                    webRtcCall = WebRtcCall(this, WebRtcCall.Role.ANSWERER, remoteVideo) { msg -> log(msg) }
                    webRtcCall?.connect(host, videoPort)
                } else if (mode == ControlMode.MANUAL || mode == ControlMode.SMART_FOLLOW || mode == ControlMode.PATROL) {
                    webRtcCall?.release()
                    webRtcCall = null
                }
            } catch (e: Exception) {
                log("Mode send failed: ${e.message}")
            }
        }
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
            Manifest.permission.RECORD_AUDIO
        )
        if (android.os.Build.VERSION.SDK_INT >= 31) {
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
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
        runOnUiThread {
            if (::logView.isInitialized) {
                logView.append("$message\n")
            } else {
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        stopReceiverServices()
        controlClient.close()
        bluetooth.close()
        webRtcCall?.release()
        senderExecutor.shutdownNow()
        super.onDestroy()
    }
}
