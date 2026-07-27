package com.cruisecar.app

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import java.io.OutputStream
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class BluetoothSppClient {
    private var socket: BluetoothSocket? = null
    private var output: OutputStream? = null

    @SuppressLint("MissingPermission")
    fun connect(address: String, onLog: (String) -> Unit) {
        close()
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: throw IllegalStateException("Bluetooth unavailable")
        val device = adapter.getRemoteDevice(address)
        socket = device.createRfcommSocketToServiceRecord(SPP_UUID)
        adapter.cancelDiscovery()
        socket?.connect()
        output = socket?.outputStream
        onLog("Bluetooth SPP connected: $address")
    }

    @SuppressLint("MissingPermission")
    fun connectFirstByName(
        context: Context,
        targetName: String = ESP32_DEVICE_NAME,
        discoveryTimeoutMs: Long = 12000,
        onLog: (String) -> Unit
    ) {
        close()
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: throw IllegalStateException("Bluetooth unavailable")

        adapter.bondedDevices
            ?.firstOrNull { it.name == targetName }
            ?.let { device ->
                onLog("Found paired ESP32: ${device.name} ${device.address}")
                connectDevice(adapter, device, onLog)
                return
            }

        onLog("Scanning Bluetooth devices for $targetName")
        val found = discoverByName(context.applicationContext, adapter, targetName, discoveryTimeoutMs, onLog)
            ?: throw IllegalStateException("$targetName not found")
        connectDevice(adapter, found, onLog)
    }

    @SuppressLint("MissingPermission")
    private fun discoverByName(
        context: Context,
        adapter: BluetoothAdapter,
        targetName: String,
        timeoutMs: Long,
        onLog: (String) -> Unit
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
                        val name = device.name ?: return
                        onLog("Bluetooth found: $name ${device.address}")
                        if (name == targetName) {
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
        socket = device.createRfcommSocketToServiceRecord(SPP_UUID)
        adapter.cancelDiscovery()
        socket?.connect()
        output = socket?.outputStream
        onLog("Bluetooth SPP connected: ${device.name ?: device.address}")
    }

    fun send(packet: ByteArray) {
        output?.write(packet)
        output?.flush()
    }

    fun isConnected(): Boolean = socket?.isConnected == true && output != null

    fun close() {
        output = null
        socket?.close()
        socket = null
    }

    companion object {
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
