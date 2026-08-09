package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import java.util.concurrent.Executors

/**
 * 独立 Debug 入口：整合 YOLO 物体识别 Demo + ESP32 蓝牙连接测试 + 小车控制。
 *
 * 通过 AndroidManifest 的独立 LAUNCHER intent-filter 与主 Activity 并列，
 * 桌面会多出一个 "CruiseCar-Debug" 图标，方便开发时快速进入调试界面。
 */
class DebugActivity : Activity() {
    private val tag = "DebugActivity"
    private val bluetooth = BluetoothSppClient()
    private val controlExecutor = Executors.newSingleThreadExecutor()
    private val blePairing = Esp32BlePairing(
        this, bluetooth,
        onLog = { log(it) },
        onToast = { toast(it) },
        onConnected = { updateEspStatus(true) }
    )

    // ---- YOLO ----
    private var yoloController: ObjectRecognitionDemoController? = null
    private var yoloPreview: CameraPreviewView? = null
    private var yoloOverlay: ObjectRecognitionOverlayView? = null
    private var yoloStatus: TextView? = null

    // ---- ESP32 ----
    private var espStatusView: TextView? = null

    private var logView: TextView? = null

    // ---- 遥控节流 ----
    private var lastGamepadState: GamepadState? = null
    private var lastGamepadAtMs: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        log("DebugActivity started")
        requestPermissions()
        buildUi()
    }

    // ---------------------------------------------------------------
    //  UI
    // ---------------------------------------------------------------

    private fun buildUi() {
        val root = vbox()
        root.addView(title("CruiseCar Debug: YOLO + ESP32"))

        // ---------- 识别 + 遥控（同一容器，复用 VideoGamepadView，与发送端一致）----------
        val previewFrame = FrameLayout(this)
        val preview = CameraPreviewView(this)
        val overlay = ObjectRecognitionOverlayView(this)
        yoloPreview = preview
        yoloOverlay = overlay
        previewFrame.addView(preview, FrameLayout.LayoutParams(-1, -1))
        previewFrame.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        val videoGamepad = VideoGamepadView(this)
        videoGamepad.setBackground(previewFrame)
        videoGamepad.onStateChanged = { state -> sendGamepadState(state) }
        root.addView(videoGamepad, LinearLayout.LayoutParams(-1, dp(420)))

        // ---------- 遥控设置(全局生效: 发送端/接收端也受此限制) ----------
        root.addView(subtitle("遥控设置"))
        val speedLabel = TextView(this).apply {
            text = "最大速度: ${CarSettings.maxSpeedPercent}%"
            textSize = 14f
            setTextColor(Color.WHITE)
        }
        root.addView(speedLabel)
        val speedSeek = SeekBar(this).apply {
            max = 100
            progress = CarSettings.maxSpeedPercent
        }
        speedSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                CarSettings.maxSpeedPercent = maxOf(progress, 10)
                speedLabel.text = "最大速度: ${CarSettings.maxSpeedPercent}%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        root.addView(speedSeek, LinearLayout.LayoutParams(-1, LinearLayout.LayoutParams.WRAP_CONTENT))

        yoloStatus = textLabel("YOLO idle")
        root.addView(yoloStatus)

        val yoloRow = hbox()
        yoloRow.addView(button("开始识别") { startYolo() }, weightParams())
        yoloRow.addView(button("停止识别") { stopYolo() }, weightParams())
        root.addView(yoloRow)

        // ---------- ESP32 connection ----------
        root.addView(separator())
        root.addView(subtitle("ESP32 连接"))

        espStatusView = textLabel("未连接").apply {
            setTextColor(Color.rgb(200, 60, 60))
        }
        root.addView(espStatusView)

        root.addView(button("自动扫描连接") { connectEsp32ByScan() })
        root.addView(button("BLE 扫描附近设备") { blePairing.start() })
        root.addView(button("断开 ESP32") { disconnectEsp32() })

        // ---------- 舵机控制 ----------
        root.addView(subtitle("舵机控制 (GPIO18)"))
        val servoLabel = TextView(this).apply {
            text = "舵机角度: 90°"
            textSize = 14f
            setTextColor(Color.WHITE)
        }
        root.addView(servoLabel)
        val servoSeek = SeekBar(this).apply {
            max = 180
            progress = 90
        }
        servoSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                servoLabel.text = "舵机角度: $progress°"
                if (fromUser) sendServo(progress)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        root.addView(servoSeek, LinearLayout.LayoutParams(-1, LinearLayout.LayoutParams.WRAP_CONTENT))

        // ---------- log ----------
        root.addView(separator())
        root.addView(subtitle("日志"))
        logView = TextView(this).apply { textSize = 12f }
        val scroll = ScrollView(this).apply { addView(logView) }
        root.addView(scroll, LinearLayout.LayoutParams(-1, dp(200)))

        val scrollRoot = ScrollView(this)
        scrollRoot.addView(root)
        setContentView(scrollRoot)
        yoloPreview?.start()
    }

    // ---------------------------------------------------------------
    //  YOLO
    // ---------------------------------------------------------------

    private fun startYolo() {
        yoloController?.stop()
        yoloController = ObjectRecognitionDemoController(
            context = this,
            frameProvider = { yoloPreview?.snapshot(416, 416) },
            onDetections = { detections ->
                runOnUiThread {
                    yoloOverlay?.setDetections(detections)
                    yoloStatus?.text = if (detections.isNotEmpty())
                        "YOLO: ${detections.size} object(s) - ${detections.joinToString { "${it.label} ${(it.confidence * 100).toInt()}%" }}"
                    else
                        "YOLO running (no detection)"
                }
            }
        ).also { it.start { msg -> log(msg) } }
    }

    private fun stopYolo() {
        yoloController?.stop()
        yoloController = null
        yoloOverlay?.setDetections(emptyList())
        yoloStatus?.text = "YOLO stopped"
    }

    // ---------------------------------------------------------------
    //  ESP32 连接
    // ---------------------------------------------------------------

    private fun connectEsp32ByScan() {
        Thread {
            try {
                log("开始扫描 ESP32 设备...")
                bluetooth.connectFirstByName(this, BluetoothSppClient.ESP32_DEVICE_NAME, 12000) { msg -> log(msg) }
                log("ESP32 连接成功")
                runOnUiThread { updateEspStatus(true) }
            } catch (e: Exception) {
                log("ESP32 扫描连接失败: ${e.message}")
                runOnUiThread { updateEspStatus(false) }
            }
        }.start()
    }

    private fun disconnectEsp32() {
        try {
            bluetooth.close()
            log("ESP32 已断开")
            runOnUiThread { updateEspStatus(false) }
        } catch (e: Exception) {
            log("断开失败: ${e.message}")
        }
    }

    private fun updateEspStatus(connected: Boolean) {
        espStatusView?.apply {
            text = if (connected) "已连接" else "未连接"
            setTextColor(if (connected) Color.rgb(60, 180, 60) else Color.rgb(200, 60, 60))
        }
    }

    // ---------------------------------------------------------------
    //  ESP32 控制
    // ---------------------------------------------------------------

    private fun sendGamepadState(state: GamepadState, force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && state == lastGamepadState && now - lastGamepadAtMs < 80) return
        if (!force && now - lastGamepadAtMs < 24) return
        lastGamepadState = state
        lastGamepadAtMs = now

        if (!bluetooth.isConnected()) {
            toast("请先连接 ESP32")
            updateEspStatus(false)
            return
        }
        controlExecutor.execute {
            try {
                val packet = state.toPacket()
                bluetooth.send(packet)
                log("发送: lx=${state.lx} ly=${state.ly} → ${packet.toHexLine()}")
            } catch (e: Exception) {
                log("发送失败: ${e.message}")
            }
        }
    }

    private fun sendServo(angle: Int) {
        if (!bluetooth.isConnected()) {
            toast("请先连接 ESP32")
            updateEspStatus(false)
            return
        }
        controlExecutor.execute {
            try {
                val packet = ServoCommand.packet(angle = angle)
                bluetooth.send(packet)
                log("舵机: angle=$angle → ${packet.toHexLine()}")
            } catch (e: Exception) {
                log("舵机发送失败: ${e.message}")
            }
        }
    }

    // ---------------------------------------------------------------
    //  Permission
    // ---------------------------------------------------------------

    private fun requestPermissions() {
        val perms = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_FINE_LOCATION,
        )
        if (android.os.Build.VERSION.SDK_INT >= 31) {
            perms.add(Manifest.permission.BLUETOOTH_CONNECT)
            perms.add(Manifest.permission.BLUETOOTH_SCAN)
        }
        val missing = perms.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 200)
        }
    }

    // ---------------------------------------------------------------
    //  UI helpers
    // ---------------------------------------------------------------

    private fun vbox(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(24, 24, 24, 24)
    }

    private fun hbox(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
    }

    private fun title(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 22f
        gravity = Gravity.CENTER_HORIZONTAL
        setTextColor(Color.rgb(58, 151, 212))
        setPadding(0, 0, 0, 12)
    }

    private fun subtitle(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 16f
        setTextColor(Color.rgb(180, 180, 180))
        setPadding(0, 8, 0, 4)
    }

    private fun separator(): View = View(this).apply {
        setBackgroundColor(Color.rgb(60, 60, 70))
        layoutParams = LinearLayout.LayoutParams(-1, 1).apply { setMargins(0, 8, 0, 8) }
    }

    private fun button(text: String, onClick: () -> Unit): Button = Button(this).apply {
        this.text = text
        setOnClickListener { onClick() }
    }

    private fun textLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 14f
        gravity = Gravity.CENTER_HORIZONTAL
        setTextColor(Color.WHITE)
    }

    private fun weightParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    // ---------------------------------------------------------------
    //  Logging
    // ---------------------------------------------------------------

    private fun log(message: String) {
        Log.i(tag, message)
        runOnUiThread {
            logView?.let {
                it.append("$message\n")
                // auto-scroll
                (it.parent as? ScrollView)?.post {
                    (it.parent as ScrollView).fullScroll(View.FOCUS_DOWN)
                }
            }
        }
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }

    // ---------------------------------------------------------------
    //  Lifecycle
    // ---------------------------------------------------------------

    override fun onDestroy() {
        blePairing.stop()
        stopYolo()
        yoloPreview?.stop()
        yoloPreview = null
        yoloOverlay = null
        yoloStatus = null
        bluetooth.close()
        controlExecutor.shutdownNow()
        super.onDestroy()
    }
}
