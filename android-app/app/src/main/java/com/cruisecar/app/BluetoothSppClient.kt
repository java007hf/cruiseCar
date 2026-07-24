package com.cruisecar.app

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothSocket
import java.io.OutputStream
import java.util.UUID

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
        val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }
}

