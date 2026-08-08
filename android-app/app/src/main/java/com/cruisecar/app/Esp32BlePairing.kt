package com.cruisecar.app

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper

/**
 * 复用组件：BLE 扫描发现 CruiseCar-ESP32，弹出设备列表供选择，
 * 选中后通过 Classic BT SPP 连接。调试台与接收端共用，保证配对方式一致。
 */
class Esp32BlePairing(
    private val activity: Activity,
    private val bluetooth: BluetoothSppClient,
    private val onLog: (String) -> Unit,
    private val onToast: (String) -> Unit,
    private val onConnected: () -> Unit
) {
    private val targetName = BluetoothSppClient.ESP32_DEVICE_NAME
    private val devices = LinkedHashMap<String, BluetoothDevice>() // addr → device
    private var scanning = false
    private var scanner: android.bluetooth.le.BluetoothLeScanner? = null
    private var timeoutRunnable: Runnable? = null
    private val handler = Handler(Looper.getMainLooper())

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val dev = result.device
            val name = dev.name ?: return
            if (!name.contains(targetName, ignoreCase = true)) return
            devices[dev.address] = dev
            onLog("BLE 发现: $name  RSSI:${result.rssi}dBm  ${dev.address}")
        }

        override fun onScanFailed(errorCode: Int) {
            onLog("BLE 扫描失败, error=$errorCode")
            stop()
        }
    }

    fun start() {
        if (scanning) {
            onToast("正在扫描中，请稍候...")
            return
        }
        val adapter = BluetoothAdapter.getDefaultAdapter()
        if (adapter == null || !adapter.isEnabled) {
            onToast("请先打开蓝牙")
            return
        }
        scanner = adapter.bluetoothLeScanner
        if (scanner == null) {
            onToast("设备不支持 BLE")
            return
        }
        // 检查权限（Android 12+ 需要 BLUETOOTH_SCAN）
        if (Build.VERSION.SDK_INT >= 31) {
            if (activity.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) {
                onToast("缺少 BLUETOOTH_SCAN 权限")
                return
            }
        } else {
            if (activity.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                onToast("缺少位置权限，无法扫描 BLE 设备")
                return
            }
        }
        devices.clear()
        scanning = true
        onLog("BLE 开始扫描（8秒超时）...")
        scanner!!.startScan(scanCallback)
        // 8 秒后自动停止并弹出设备列表
        timeoutRunnable = Runnable {
            stop()
            showPicker()
        }
        handler.postDelayed(timeoutRunnable!!, 8000)
    }

    fun stop() {
        if (!scanning) return
        scanning = false
        timeoutRunnable?.let { handler.removeCallbacks(it) }
        timeoutRunnable = null
        try {
            if (Build.VERSION.SDK_INT >= 31) {
                if (activity.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED)
                    scanner?.stopScan(scanCallback)
            } else {
                scanner?.stopScan(scanCallback)
            }
        } catch (_: Exception) {
        }
        onLog("BLE 扫描结束，发现 ${devices.size} 台可连接设备")
    }

    private fun showPicker() {
        if (devices.isEmpty()) {
            onToast("未发现 CruiseCar-ESP32 设备")
            return
        }
        val entries = devices.values.toList()
        val labels = entries.map { "${it.name}  (${it.address})" }.toTypedArray()
        AlertDialog.Builder(activity)
            .setTitle("选择 ESP32 设备 (${entries.size}台)")
            .setItems(labels) { _, which ->
                val dev = entries[which]
                onLog("用户选择: ${dev.name}  ${dev.address}")
                connect(dev)
            }
            .setNegativeButton("取消") { dialog, _ -> dialog.dismiss() }
            .show()
    }

    private fun connect(device: BluetoothDevice) {
        Thread {
            try {
                val addr = device.address
                onLog("通过 BLE 发现 → Classic BT SPP 连接: $addr")
                bluetooth.connect(addr) { msg -> onLog(msg) }
                onLog("ESP32 SPP 连接成功: $addr")
                activity.runOnUiThread { onConnected() }
            } catch (e: Exception) {
                onLog("ESP32 连接失败 (BLE发现后): ${e.message}")
            }
        }.start()
    }
}
