package com.cruisecar.app.protocol

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicInteger

data class DebugTraceFrame(
    val seq: Int,
    val packet: ByteArray,
    val timestamps: LongArray
)

object DebugTracePacket {
    const val TYPE_TRACE = 0xF0
    const val TIMESTAMP_COUNT = 8
    const val STAGE_SENDER_CREATED = 0
    const val STAGE_SENDER_TCP_WRITE = 1
    const val STAGE_SERVER_RECEIVED = 2
    const val STAGE_SERVER_FORWARD = 3
    const val STAGE_RECEIVER_TCP_RECEIVED = 4
    const val STAGE_RECEIVER_BT_ENQUEUE = 5
    const val STAGE_RECEIVER_BT_WRITE = 6
    private const val VERSION = 1
    private const val PAYLOAD_SIZE = 1 + 4 + ControlProtocol.PACKET_SIZE + TIMESTAMP_COUNT * 8
    const val FRAME_SIZE = 4 + PAYLOAD_SIZE + 1
    private val seq = AtomicInteger(1)

    fun wrap(packet: ByteArray, timestampStage: Int? = null, nowMs: Long = System.currentTimeMillis()): ByteArray {
        val timestamps = LongArray(TIMESTAMP_COUNT)
        if (timestampStage != null && timestampStage in timestamps.indices) {
            timestamps[timestampStage] = nowMs
        }
        return encode(seq.getAndIncrement(), packet, timestamps)
    }

    fun mark(transport: ByteArray, stage: Int, nowMs: Long = System.currentTimeMillis()): ByteArray {
        if (!isTrace(transport)) {
            return wrap(transport, stage, nowMs)
        }
        val frame = parse(transport) ?: return transport
        val timestamps = frame.timestamps.copyOf()
        if (stage in timestamps.indices) {
            timestamps[stage] = nowMs
        }
        return encode(frame.seq, frame.packet, timestamps)
    }

    fun controlPacket(transport: ByteArray): ByteArray =
        parse(transport)?.packet ?: transport

    fun parse(transport: ByteArray): DebugTraceFrame? {
        if (!isTrace(transport) || transport.size != FRAME_SIZE) return null
        val payloadSize = transport[3].toInt() and 0xFF
        if (payloadSize != PAYLOAD_SIZE) return null
        if (checksum(transport, transport.size - 1) != transport.last()) return null
        val buffer = ByteBuffer.wrap(transport, 4, payloadSize).order(ByteOrder.LITTLE_ENDIAN)
        val version = buffer.get().toInt() and 0xFF
        if (version != VERSION) return null
        val seq = buffer.getInt()
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        buffer.get(packet)
        val timestamps = LongArray(TIMESTAMP_COUNT) { buffer.getLong() }
        return DebugTraceFrame(seq, packet, timestamps)
    }

    fun isTrace(data: ByteArray): Boolean =
        data.size >= 4 &&
            (data[0].toInt() and 0xFF) == ControlProtocol.HEADER_0 &&
            (data[1].toInt() and 0xFF) == ControlProtocol.HEADER_1 &&
            (data[2].toInt() and 0xFF) == TYPE_TRACE

    private fun encode(seq: Int, packet: ByteArray, timestamps: LongArray): ByteArray {
        require(packet.size == ControlProtocol.PACKET_SIZE) { "debug trace must wrap a 10-byte control packet" }
        val out = ByteArray(FRAME_SIZE)
        out[0] = ControlProtocol.HEADER_0.toByte()
        out[1] = ControlProtocol.HEADER_1.toByte()
        out[2] = TYPE_TRACE.toByte()
        out[3] = PAYLOAD_SIZE.toByte()
        val buffer = ByteBuffer.wrap(out, 4, PAYLOAD_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(VERSION.toByte())
        buffer.putInt(seq)
        buffer.put(packet)
        for (i in 0 until TIMESTAMP_COUNT) {
            buffer.putLong(timestamps.getOrElse(i) { 0L })
        }
        out[out.lastIndex] = checksum(out, out.lastIndex)
        return out
    }

    private fun checksum(data: ByteArray, count: Int): Byte {
        var sum = 0
        for (i in 0 until count) {
            sum = (sum + (data[i].toInt() and 0xFF)) and 0xFF
        }
        return sum.toByte()
    }
}
