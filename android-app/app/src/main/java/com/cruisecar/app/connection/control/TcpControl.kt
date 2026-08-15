package com.cruisecar.app.connection.control

import com.cruisecar.app.BuildConfig
import com.cruisecar.app.protocol.ControlCommand
import com.cruisecar.app.protocol.ControlFrame
import com.cruisecar.app.protocol.ControlMode
import com.cruisecar.app.protocol.ControlProtocol
import com.cruisecar.app.protocol.DebugTracePacket
import com.cruisecar.app.protocol.GamepadState
import com.cruisecar.app.protocol.ModeCommand
import com.cruisecar.app.protocol.ServoCommand
import com.cruisecar.app.protocol.toHexLine
import java.io.BufferedInputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class ControlServer(private val port: Int, private val onFrame: (ByteArray, ControlFrame) -> Unit) {
    private val running = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null
    private val clientOutputs = mutableListOf<OutputStream>()
    private val clientLock = Any()

    fun start(onLog: (String) -> Unit) {
        if (!running.compareAndSet(false, true)) return
        Thread {
            try {
                serverSocket = ServerSocket(port)
                onLog("TCP control server started on $port")
                while (running.get()) {
                    val client = serverSocket?.accept() ?: break
                    onLog("Control client connected: ${client.inetAddress.hostAddress}")
                    Thread { readClient(client, onLog) }.start()
                }
            } catch (ignored: Exception) {
                if (running.get()) onLog("Control server stopped unexpectedly")
            }
        }.start()
    }

    private fun readClient(client: Socket, onLog: (String) -> Unit) {
        val output = client.getOutputStream()
        synchronized(clientLock) { clientOutputs.add(output) }
        try {
            client.use {
                val input = BufferedInputStream(it.getInputStream())
                while (running.get()) {
                    val transport = readControlTransport(input) ?: break
                    if (!running.get()) break
                    val packet = DebugTracePacket.controlPacket(transport)
                    val parsed = ControlProtocol.parse(packet)
                    if (parsed != null) {
                        try {
                            val stamped = if (DebugTracePacket.isTrace(transport)) {
                                DebugTracePacket.mark(transport, DebugTracePacket.STAGE_RECEIVER_TCP_RECEIVED)
                            } else {
                                transport
                            }
                            onFrame(stamped, parsed)
                        } catch (e: Exception) {
                            onLog("Frame handling error: ${e.message}")
                        }
                    } else {
                        onLog("Dropped invalid control frame: ${transport.toHexLine()}")
                    }
                }
            }
        } catch (e: Exception) {
            onLog("Control client read ended: ${e.message}")
        } finally {
            synchronized(clientLock) { clientOutputs.remove(output) }
            onLog("Control client disconnected")
        }
    }

    /** 向所有已连接的发送端回传数据(如状态回报)。 */
    fun send(packet: ByteArray) {
        synchronized(clientLock) {
            val it = clientOutputs.iterator()
            while (it.hasNext()) {
                val out = it.next()
                try {
                    out.write(packet)
                    out.flush()
                } catch (e: Exception) {
                    it.remove()
                }
            }
        }
    }

    fun stop() {
        running.set(false)
        serverSocket?.close()
        serverSocket = null
        synchronized(clientLock) { clientOutputs.clear() }
    }
}

class ControlClient {
    private val running = AtomicBoolean(false)
    private var socket: Socket? = null
    private var output: OutputStream? = null
    private var readThread: Thread? = null
    private var skipHandshakeAck = false
    /** 连接世代计数: 每次 connect/close 递增, 用于丢弃断连期间排队的过期发送任务。 */
    private val connectionEpoch = AtomicInteger(0)

    var onStatus: ((Boolean, ControlMode) -> Unit)? = null
    var onReceiverGone: (() -> Unit)? = null
    var onFrame: ((ByteArray, ControlFrame) -> Unit)? = null
    var debugTraceEnabled: Boolean = BuildConfig.DEBUG

    fun connect(host: String, port: Int, onLog: (String) -> Unit) {
        close()
        socket = Socket(host, port)
        output = socket?.getOutputStream()
        skipHandshakeAck = false
        running.set(true)
        startReadLoop()
        onLog("Connected to receiver $host:$port")
    }

    fun connectRemoteSender(host: String, port: Int, senderId: String, targetDeviceId: String, token: String, onLog: (String) -> Unit) {
        connectWithHandshake(
            host,
            port,
            "{\"role\":\"sender\",\"sender_id\":\"${senderId.jsonEscape()}\",\"target_device_id\":\"${targetDeviceId.jsonEscape()}\",\"token\":\"${token.jsonEscape()}\"}\n",
            onLog
        )
    }

    fun connectRemoteReceiver(host: String, port: Int, deviceId: String, token: String, onLog: (String) -> Unit) {
        connectWithHandshake(
            host,
            port,
            "{\"role\":\"receiver\",\"device_id\":\"${deviceId.jsonEscape()}\",\"token\":\"${token.jsonEscape()}\"}\n",
            onLog
        )
    }

    private fun connectWithHandshake(host: String, port: Int, hello: String, onLog: (String) -> Unit) {
        close()
        socket = Socket(host, port)
        output = socket?.getOutputStream()
        output?.write(hello.toByteArray(Charsets.UTF_8))
        output?.flush()
        skipHandshakeAck = true
        running.set(true)
        startReadLoop()
        onLog("Connected to remote server $host:$port")
    }

    private fun startReadLoop() {
        val input = socket?.getInputStream() ?: return
        readThread = Thread {
            try {
                if (skipHandshakeAck) {
                    while (true) {
                        val b = input.read()
                        if (b < 0 || b == '\n'.code) break
                    }
                }
                while (running.get() && socket?.isConnected == true) {
                    val transport = readControlTransport(input) ?: break
                    if (!running.get()) break
                    val packet = DebugTracePacket.controlPacket(transport)
                    val parsed = ControlProtocol.parse(packet)
                    if (parsed is ControlFrame.Status) {
                        onStatus?.invoke(parsed.espConnected, parsed.mode)
                    } else if (parsed != null) {
                        val stamped = if (DebugTracePacket.isTrace(transport)) {
                            DebugTracePacket.mark(transport, DebugTracePacket.STAGE_RECEIVER_TCP_RECEIVED)
                        } else {
                            transport
                        }
                        onFrame?.invoke(stamped, parsed)
                    }
                }
            } catch (ignored: Exception) {
            }
            running.set(false)
            onReceiverGone?.invoke()
        }.also { it.start() }
    }

    @Synchronized
    fun send(state: GamepadState, createdAtMs: Long = System.currentTimeMillis()) {
        sendPacket(state.toPacket(), traceStage = DebugTracePacket.STAGE_SENDER_CREATED, traceAtMs = createdAtMs)
    }

    @Synchronized
    fun sendMode(mode: ControlMode) {
        sendPacket(ModeCommand.packet(mode), traceStage = DebugTracePacket.STAGE_SENDER_CREATED)
    }

    @Synchronized
    fun sendServo(index: Int = 0, angle: Int) {
        sendPacket(ServoCommand.packet(index, angle), traceStage = DebugTracePacket.STAGE_SENDER_CREATED)
    }

    @Synchronized
    fun sendCommand(code: Int) {
        sendPacket(ControlCommand.packet(code), traceStage = DebugTracePacket.STAGE_SENDER_CREATED)
    }

    @Synchronized
    fun sendRaw(packet: ByteArray) {
        output?.write(packet)
        output?.flush()
    }

    private fun sendPacket(packet: ByteArray, traceStage: Int, traceAtMs: Long = System.currentTimeMillis()) {
        val out = output ?: return
        val transport = if (debugTraceEnabled) {
            DebugTracePacket.wrap(packet, traceStage, traceAtMs)
        } else {
            packet
        }
        val stamped = if (DebugTracePacket.isTrace(transport)) {
            DebugTracePacket.mark(transport, DebugTracePacket.STAGE_SENDER_TCP_WRITE)
        } else {
            transport
        }
        out.write(stamped)
        out.flush()
    }

    fun isConnected(): Boolean = socket?.isConnected == true && output != null && running.get()

    /** 当前连接世代; 发送任务应在提交时记录该值, 执行时比对, 不一致即说明连接已变更。 */
    fun connectionEpoch(): Int = connectionEpoch.get()

    fun close() {
        running.set(false)
        output = null
        skipHandshakeAck = false
        socket?.close()
        socket = null
        readThread = null
        connectionEpoch.incrementAndGet()
    }
}

private fun readControlTransport(input: InputStream): ByteArray? {
    while (true) {
        val first = input.read()
        if (first < 0) return null
        if (first != ControlProtocol.HEADER_0) continue
        val second = input.read()
        if (second < 0) return null
        if (second != ControlProtocol.HEADER_1) continue
        val type = input.read()
        if (type < 0) return null
        if (type == DebugTracePacket.TYPE_TRACE) {
            val payloadSize = input.read()
            if (payloadSize < 0) return null
            val frame = ByteArray(4 + payloadSize + 1)
            frame[0] = first.toByte()
            frame[1] = second.toByte()
            frame[2] = type.toByte()
            frame[3] = payloadSize.toByte()
            if (!readFully(input, frame, 4, payloadSize + 1)) return null
            return frame
        }
        val frame = ByteArray(ControlProtocol.PACKET_SIZE)
        frame[0] = first.toByte()
        frame[1] = second.toByte()
        frame[2] = type.toByte()
        if (!readFully(input, frame, 3, ControlProtocol.PACKET_SIZE - 3)) return null
        return frame
    }
}

private fun readFully(input: InputStream, out: ByteArray, offset: Int, length: Int): Boolean {
    var cursor = offset
    val end = offset + length
    while (cursor < end) {
        val read = input.read(out, cursor, end - cursor)
        if (read < 0) return false
        cursor += read
    }
    return true
}

private fun String.jsonEscape(): String =
    replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
