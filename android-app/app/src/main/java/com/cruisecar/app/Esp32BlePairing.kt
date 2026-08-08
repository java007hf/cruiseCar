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
import android.util.Log

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
    private val TAG = "Esp32BlePairing"
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
            val addr = device.address
            val typeStr = when (device.type) {
                BluetoothDevice.DEVICE_TYPE_CLASSIC -> "CLASSIC"
                BluetoothDevice.DEVICE_TYPE_DUAL -> "DUAL"
                BluetoothDevice.DEVICE_TYPE_LE -> "LE"
                BluetoothDevice.DEVICE_TYPE_UNKNOWN -> "UNKNOWN"
                else -> "?${device.type}"
            }
            val bondStr = when (device.bondState) {
                BluetoothDevice.BOND_BONDED -> "BONDED"
                BluetoothDevice.BOND_BONDING -> "BONDING"
                BluetoothDevice.BOND_NONE -> "NONE"
                else -> "?${device.bondState}"
            }
            onLog("选中设备: $addr name=${device.name} type=$typeStr bond=$bondStr")
            Log.d(TAG, "connect() selected addr=$addr type=$typeStr bond=$bondStr")
            try {
                onLog("连接选中的设备 (Classic SPP): $addr")
                bluetooth.connect(addr) { msg -> onLog(msg) }
                onLog("ESP32 SPP 连接成功: $addr")
                activity.runOnUiThread { onConnected() }
                return@Thread
            } catch (e: Exception) {
                Log.e(TAG, "直连选中设备失败，改按同地址 Classic 扫描", e)
                onLog("直连失败 (${e.message})，改用 Classic 按同一地址重连...")
            }
            // 兜底：按"同一地址"做 Classic 扫描后连接，绝不连到别的/旧的同名已配对设备。
            try {
                val name = device.name ?: BluetoothSppClient.ESP32_DEVICE_NAME
                bluetooth.connectByClassicScan(activity, addr, name) { msg -> onLog(msg) }
                onLog("ESP32 SPP 连接成功 (同地址 Classic): $addr")
                activity.runOnUiThread { onConnected() }
            } catch (e2: Exception) {
                Log.e(TAG, "同地址 Classic 重连也失败", e2)
                onLog("连接失败 (${e2.message})，请重新扫描并选择设备")
            }
        }.start()
    }
}
