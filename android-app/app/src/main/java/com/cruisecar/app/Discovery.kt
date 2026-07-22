package com.cruisecar.app

import android.net.wifi.WifiManager
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.SocketTimeoutException
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean

data class ReceiverInfo(val name: String, val host: String, val port: Int)

class DiscoveryResponder(private val tcpPort: Int) {
    private val running = AtomicBoolean(false)
    private var socket: DatagramSocket? = null

    fun start(onLog: (String) -> Unit) {
        if (!running.compareAndSet(false, true)) return
        Thread {
            try {
                socket = DatagramSocket(DISCOVERY_PORT).apply { broadcast = true }
                val buffer = ByteArray(512)
                onLog("UDP discovery responder started on $DISCOVERY_PORT")
                while (running.get()) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket?.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    if (message == DISCOVERY_REQUEST) {
                        val response = "$DISCOVERY_RESPONSE|AndroidReceiver|$tcpPort".toByteArray()
                        val reply = DatagramPacket(response, response.size, packet.address, packet.port)
                        socket?.send(reply)
                    }
                }
            } catch (ignored: Exception) {
                if (running.get()) onLog("Discovery responder stopped unexpectedly")
            }
        }.start()
    }

    fun stop() {
        running.set(false)
        socket?.close()
        socket = null
    }
}

class DiscoveryScanner(private val wifiManager: WifiManager?) {
    fun scan(timeoutMs: Int = 2500, onLog: (String) -> Unit): List<ReceiverInfo> {
        val lock = wifiManager?.createMulticastLock("cruisecar-discovery")
        val found = CopyOnWriteArrayList<ReceiverInfo>()
        lock?.setReferenceCounted(false)
        lock?.acquire()
        DatagramSocket(null).use { socket ->
            socket.reuseAddress = true
            socket.broadcast = true
            socket.soTimeout = timeoutMs
            socket.bind(InetSocketAddress(0))
            val bytes = DISCOVERY_REQUEST.toByteArray()
            socket.send(DatagramPacket(bytes, bytes.size, InetAddress.getByName("255.255.255.255"), DISCOVERY_PORT))
            val start = System.currentTimeMillis()
            val buffer = ByteArray(512)
            while (System.currentTimeMillis() - start < timeoutMs) {
                try {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    val parts = message.split("|")
                    if (parts.size == 3 && parts[0] == DISCOVERY_RESPONSE) {
                        found.add(ReceiverInfo(parts[1], packet.address.hostAddress ?: "", parts[2].toInt()))
                    }
                } catch (_: SocketTimeoutException) {
                    break
                }
            }
        }
        lock?.release()
        onLog("Found ${found.size} receiver(s)")
        return found.distinctBy { "${it.host}:${it.port}" }
    }

    companion object {
        const val DISCOVERY_PORT = 42100
        const val DISCOVERY_REQUEST = "CRUISE_CAR_DISCOVER_V1"
        const val DISCOVERY_RESPONSE = "CRUISE_CAR_RECEIVER_V1"
    }
}

private const val DISCOVERY_PORT = DiscoveryScanner.DISCOVERY_PORT
private const val DISCOVERY_REQUEST = DiscoveryScanner.DISCOVERY_REQUEST
private const val DISCOVERY_RESPONSE = DiscoveryScanner.DISCOVERY_RESPONSE

