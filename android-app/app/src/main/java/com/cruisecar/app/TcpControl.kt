package com.cruisecar.app

import java.io.BufferedInputStream
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

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
                val frame = ByteArray(ControlProtocol.PACKET_SIZE)
                while (running.get()) {
                    var offset = 0
                    var eof = false
                    while (offset < frame.size) {
                        val read = input.read(frame, offset, frame.size - offset)
                        if (read < 0) { eof = true; break }
                        offset += read
                    }
                    if (eof || !running.get()) break
                    val packet = frame.copyOf()
                    val parsed = ControlProtocol.parse(packet)
                    if (parsed != null) {
                        try {
                            onFrame(packet, parsed)
                        } catch (e: Exception) {
                            onLog("Frame handling error: ${e.message}")
                        }
                    } else {
                        onLog("Dropped invalid control frame: ${packet.toHexLine()}")
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

    var onStatus: ((Boolean, ControlMode) -> Unit)? = null
    var onReceiverGone: (() -> Unit)? = null

    fun connect(host: String, port: Int, onLog: (String) -> Unit) {
        close()
        socket = Socket(host, port)
        output = socket?.getOutputStream()
        running.set(true)
        startReadLoop()
        onLog("Connected to receiver $host:$port")
    }

    private fun startReadLoop() {
        val input = socket?.getInputStream() ?: return
        readThread = Thread {
            val frame = ByteArray(ControlProtocol.PACKET_SIZE)
            try {
                while (running.get() && socket?.isConnected == true) {
                    var offset = 0
                    var eof = false
                    while (offset < frame.size) {
                        val read = input.read(frame, offset, frame.size - offset)
                        if (read < 0) { eof = true; break }
                        offset += read
                    }
                    if (eof || !running.get()) break
                    val parsed = ControlProtocol.parse(frame.copyOf())
                    if (parsed is ControlFrame.Status) {
                        onStatus?.invoke(parsed.espConnected, parsed.mode)
                    }
                }
            } catch (ignored: Exception) {
            }
            running.set(false)
            onReceiverGone?.invoke()
        }.also { it.start() }
    }

    @Synchronized
    fun send(state: GamepadState) {
        output?.write(state.toPacket())
        output?.flush()
    }

    @Synchronized
    fun sendMode(mode: ControlMode) {
        output?.write(ModeCommand.packet(mode))
        output?.flush()
    }

    @Synchronized
    fun sendServo(index: Int = 0, angle: Int) {
        output?.write(ServoCommand.packet(index, angle))
        output?.flush()
    }

    @Synchronized
    fun sendCommand(code: Int) {
        output?.write(ControlCommand.packet(code))
        output?.flush()
    }

    fun isConnected(): Boolean = socket?.isConnected == true && output != null && running.get()

    fun close() {
        running.set(false)
        output = null
        socket?.close()
        socket = null
        readThread = null
    }
}
