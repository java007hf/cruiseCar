package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
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

class MainActivity : Activity() {
    private val controlPort = 42101
    private val controlClient = ControlClient()
    private val bluetooth = BluetoothSppClient()
    private var discoveryResponder: DiscoveryResponder? = null
    private var controlServer: ControlServer? = null
    private lateinit var logView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestBasePermissions()
        showRoleScreen()
    }

    private fun showRoleScreen() {
        val layout = rootLayout()
        layout.addView(title("CruiseCar"))
        layout.addView(button("发送设备") { showSenderScreen() })
        layout.addView(button("接收设备") { showReceiverScreen() })
        setContentView(withLog(layout))
    }

    private fun showSenderScreen() {
        val layout = rootLayout()
        layout.addView(title("发送设备"))
        layout.addView(button("扫描接收设备并连接") {
            Thread {
                try {
                    val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                    val receivers = DiscoveryScanner(wifi).scan { log(it) }
                    if (receivers.isNotEmpty()) {
                        val receiver = receivers.first()
                        controlClient.connect(receiver.host, receiver.port) { log(it) }
                    }
                } catch (e: Exception) {
                    log("Scan/connect failed: ${e.message}")
                }
            }.start()
        })
        layout.addView(button("前进") { send(GamepadState(ly = 0)) })
        layout.addView(button("后退") { send(GamepadState(ly = 255)) })
        layout.addView(button("左转") { send(GamepadState(lx = 0)) })
        layout.addView(button("右转") { send(GamepadState(lx = 255)) })
        layout.addView(button("停止") { send(GamepadState()) })
        layout.addView(button("返回") { showRoleScreen() })
        setContentView(withLog(layout))
    }

    private fun showReceiverScreen() {
        val layout = rootLayout()
        layout.addView(title("接收设备"))
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
        layout.addView(button("启动接收服务") {
            startReceiverServices()
        })
        layout.addView(button("停止") {
            stopReceiverServices()
            bluetooth.close()
            log("Receiver stopped")
        })
        layout.addView(button("返回") { showRoleScreen() })
        setContentView(withLog(layout))
    }

    private fun startReceiverServices() {
        discoveryResponder = DiscoveryResponder(controlPort).also { it.start { msg -> log(msg) } }
        controlServer = ControlServer(controlPort) { packet ->
            bluetooth.send(packet)
            log("Forwarded: ${packet.joinToString(" ") { "%02X".format(it.toInt() and 0xFF) }}")
        }.also { it.start { msg -> log(msg) } }
    }

    private fun stopReceiverServices() {
        discoveryResponder?.stop()
        controlServer?.stop()
        discoveryResponder = null
        controlServer = null
    }

    private fun send(state: GamepadState) {
        Thread {
            try {
                controlClient.send(state)
                log("Sent: ${state.toHexLine()}")
            } catch (e: Exception) {
                log("Send failed: ${e.message}")
            }
        }.start()
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

    private fun title(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 28f
        gravity = Gravity.CENTER_HORIZONTAL
    }

    private fun button(text: String, onClick: () -> Unit): Button = Button(this).apply {
        this.text = text
        setOnClickListener { onClick() }
    }

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
        super.onDestroy()
    }
}

