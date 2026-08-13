package com.cruisecar.app.connection.esp32

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.util.Log
import java.io.IOException
import java.io.OutputStream
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class BluetoothSppClient {
    @Volatile private var socket: BluetoothSocket? = null
    @Volatile private var output: OutputStream? = null
    @Volatile private var connected = false

    private fun deviceTypeStr(type: Int): String = when (type) {
        BluetoothDevice.DEVICE_TYPE_CLASSIC -> "CLASSIC"
        BluetoothDevice.DEVICE_TYPE_DUAL -> "DUAL"
        BluetoothDevice.DEVICE_TYPE_LE -> "LE"
        BluetoothDevice.DEVICE_TYPE_UNKNOWN -> "UNKNOWN"
        else -> "?$type"
    }

    private fun bondStr(state: Int): String = when (state) {
        BluetoothDevice.BOND_BONDED -> "BONDED"
        BluetoothDevice.BOND_BONDING -> "BONDING"
        BluetoothDevice.BOND_NONE -> "NONE"
        else -> "?$state"
    }

    private fun dumpDevice(device: BluetoothDevice?) {
        if (device == null) {
            Log.d(TAG, "device=null")
            return
        }
        Log.d(TAG, "device addr=${device.address} name=${device.name} " +
                "type=${deviceTypeStr(device.type)} bond=${bondStr(device.bondState)}")
    }

    @SuppressLint("MissingPermission")
    fun connect(address: String, onLog: (String) -> Unit) {
        close()
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: throw IllegalStateException("Bluetooth unavailable")
        val device = adapter.getRemoteDevice(address)
        dumpDevice(device)
        onLog("SPP: 创建 insecure socket addr=$address uuid=$SPP_UUID")
        Log.d(TAG, "createInsecureRfcommSocketToServiceRecord begin")
        socket = device.createInsecureRfcommSocketToServiceRecord(SPP_UUID)
        Log.d(TAG, "socket created, cancelDiscovery...")
        onLog("SPP: socket 已创建, cancelDiscovery ...")
        adapter.cancelDiscovery()
        val t0 = System.currentTimeMillis()
        try {
            onLog("SPP: socket.connect() ...")
            Log.d(TAG, "socket.connect() begin")
            socket?.connect()
            Log.d(TAG, "socket.connect() done in ${System.currentTimeMillis() - t0}ms")
        } catch (e: Exception) {
            Log.e(TAG, "socket.connect() FAILED after ${System.currentTimeMillis() - t0}ms", e)
            onLog("SPP: connect 失败: ${e.message}")
            throw e
        }
        output = socket?.outputStream
        connected = true
        onLog("Bluetooth SPP connected: $address")
    }

    /**
     * 自动扫描连接：先扫描活着的设备(不再优先用旧配对)，
     * 找不到再退而求其次连已配对设备(可能处于不可发现模式)。
     */
    @SuppressLint("MissingPermission")
    fun connectFirstByName(
        context: Context,
        targetName: String = ESP32_DEVICE_NAME,
        discoveryTimeoutMs: Long = 12000,
        onLog: (String) -> Unit
    ) {
        close()
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: throw IllegalStateException("Bluetooth unavailable")
        onLog("Classic 扫描 $targetName 中(不再优先用旧配对)...")
        Log.d(TAG, "scan-first for '$targetName'")
        val found = discoverByName(context.applicationContext, adapter, targetName, discoveryTimeoutMs, onLog)
        if (found != null) {
            connectDevice(adapter, found, onLog)
            return
        }
        // 扫描未命中：兜底尝试已配对设备(可能处于不可发现模式)
        val bonded = adapter.bondedDevices?.firstOrNull { it.name == targetName }
        if (bonded != null) {
            dumpDevice(bonded)
            onLog("扫描未找到，回退到已配对设备: ${bonded.name} ${bonded.address}")
            Log.d(TAG, "fallback to bonded device")
            connectDevice(adapter, bonded, onLog)
            return
        }
        throw IllegalStateException("$targetName not found")
    }

    /**
     * 连接用户在 BLE 里明确选中的那台设备：用 Classic 扫描并按地址匹配，
     * 避免连到别的/旧的同名已配对设备。
     */
    @SuppressLint("MissingPermission")
    fun connectByClassicScan(
        context: Context,
        address: String,
        targetName: String = ESP32_DEVICE_NAME,
        discoveryTimeoutMs: Long = 12000,
        onLog: (String) -> Unit
    ) {
        close()
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: throw IllegalStateException("Bluetooth unavailable")
        onLog("Classic 扫描同地址 $address 中...")
        Log.d(TAG, "connectByClassicScan address=$address")
        val found = discoverByAddress(context.applicationContext, adapter, address, discoveryTimeoutMs, onLog)
            ?: throw IllegalStateException("address $address not found in Classic scan")
        connectDevice(adapter, found, onLog)
    }

    @SuppressLint("MissingPermission")
    private fun discoverByName(
        context: Context,
        adapter: BluetoothAdapter,
        targetName: String,
        timeoutMs: Long,
        onLog: (String) -> Unit
    ): BluetoothDevice? = discover(context, adapter, timeoutMs, onLog) { device ->
        device.name == targetName
    }

    @SuppressLint("MissingPermission")
    private fun discoverByAddress(
        context: Context,
        adapter: BluetoothAdapter,
        address: String,
        timeoutMs: Long,
        onLog: (String) -> Unit
    ): BluetoothDevice? = discover(context, adapter, timeoutMs, onLog) { device ->
        device.address.equals(address, ignoreCase = true)
    }

    @SuppressLint("MissingPermission")
    private fun discover(
        context: Context,
        adapter: BluetoothAdapter,
        timeoutMs: Long,
        onLog: (String) -> Unit,
        match: (BluetoothDevice) -> Boolean
    ): BluetoothDevice? {
        val found = AtomicReference<BluetoothDevice?>()
        val done = CountDownLatch(1)
        val filter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_FOUND)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
        }
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                when (intent.action) {
                    BluetoothDevice.ACTION_FOUND -> {
                        val device = intent.bluetoothDeviceExtra() ?: return
                        dumpDevice(device)
                        onLog("Classic 发现: ${device.name} ${device.address} type=${deviceTypeStr(device.type)}")
                        if (match(device)) {
                            found.set(device)
                            adapter.cancelDiscovery()
                            done.countDown()
                        }
                    }
                    BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> done.countDown()
                }
            }
        }

        if (Build.VERSION.SDK_INT >= 33) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(receiver, filter)
        }

        try {
            if (adapter.isDiscovering) adapter.cancelDiscovery()
            if (!adapter.startDiscovery()) {
                throw IllegalStateException("Bluetooth discovery failed to start")
            }
            done.await(timeoutMs, TimeUnit.MILLISECONDS)
            return found.get()
        } finally {
            runCatching { context.unregisterReceiver(receiver) }
            if (adapter.isDiscovering) adapter.cancelDiscovery()
        }
    }

    @SuppressLint("MissingPermission")
    private fun connectDevice(adapter: BluetoothAdapter, device: BluetoothDevice, onLog: (String) -> Unit) {
        dumpDevice(device)
        onLog("SPP: 创建 insecure socket addr=${device.address} uuid=$SPP_UUID")
        Log.d(TAG, "createInsecureRfcommSocketToServiceRecord begin")
        socket = device.createInsecureRfcommSocketToServiceRecord(SPP_UUID)
        Log.d(TAG, "socket created, cancelDiscovery...")
        adapter.cancelDiscovery()
        val t0 = System.currentTimeMillis()
        try {
            onLog("SPP: socket.connect() ...")
            Log.d(TAG, "socket.connect() begin")
            socket?.connect()
            Log.d(TAG, "socket.connect() done in ${System.currentTimeMillis() - t0}ms")
        } catch (e: Exception) {
            Log.e(TAG, "socket.connect() FAILED after ${System.currentTimeMillis() - t0}ms", e)
            onLog("SPP: connect 失败: ${e.message}")
            throw e
        }
        output = socket?.outputStream
        connected = true
        onLog("Bluetooth SPP connected: ${device.name ?: device.address}")
    }

    /**
     * 发送数据包，返回是否成功。
     * 失败时(蓝牙断开/写异常)会标记 connected=false 并打日志，不再向上抛 IOException，
     * 避免调用方在裸线程(如 TCP 控制服务端)上未捕获异常导致崩溃。
     */
    fun send(packet: ByteArray): Boolean {
        if (!connected) return false
        return try {
            output?.write(packet)
            output?.flush()
            true
        } catch (e: IOException) {
            Log.e(TAG, "send failed, marking disconnected: ${e.message}")
            connected = false
            false
        }
    }

    fun isConnected(): Boolean = connected

    fun close() {
        connected = false
        output = null
        socket?.close()
        socket = null
    }

    companion object {
        private const val TAG = "BtSppClient"
        const val ESP32_DEVICE_NAME = "CruiseCar-ESP32"
        val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }
}

@Suppress("DEPRECATION")
private fun Intent.bluetoothDeviceExtra(): BluetoothDevice? =
    if (Build.VERSION.SDK_INT >= 33) {
        getParcelableExtra(BluetoothDevice.EXTRA_DEVICE, BluetoothDevice::class.java)
    } else {
        getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
    }
