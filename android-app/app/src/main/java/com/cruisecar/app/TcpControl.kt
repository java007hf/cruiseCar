package com.cruisecar.app

import java.io.BufferedInputStream
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

class ControlServer(private val port: Int, private val onPacket: (ByteArray) -> Unit) {
    private val running = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null

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
        client.use {
            val input = BufferedInputStream(it.getInputStream())
            val frame = ByteArray(10)
            while (running.get()) {
                var offset = 0
                while (offset < frame.size) {
                    val read = input.read(frame, offset, frame.size - offset)
                    if (read < 0) return
                    offset += read
                }
                onPacket(frame.copyOf())
            }
        }
        onLog("Control client disconnected")
    }

    fun stop() {
        running.set(false)
        serverSocket?.close()
        serverSocket = null
    }
}

class ControlClient {
    private var socket: Socket? = null
    private var output: OutputStream? = null

    fun connect(host: String, port: Int, onLog: (String) -> Unit) {
        close()
        socket = Socket(host, port)
        output = socket?.getOutputStream()
        onLog("Connected to receiver $host:$port")
    }

    fun send(state: GamepadState) {
        output?.write(state.toPacket())
        output?.flush()
    }

    fun close() {
        output = null
        socket?.close()
        socket = null
    }
}

