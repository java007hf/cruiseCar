package com.cruisecar.app.protocol

import kotlin.math.roundToInt

sealed class ControlFrame {
    data class Gamepad(val state: GamepadState) : ControlFrame()
    data class Mode(val mode: ControlMode) : ControlFrame()
    data class Servo(val index: Int, val angle: Int) : ControlFrame()
    data class Status(val espConnected: Boolean, val mode: ControlMode) : ControlFrame()
    data class Command(val code: Int) : ControlFrame()
    data class DebugAck(val seq: Int) : ControlFrame()
}

enum class ControlMode(val wireValue: Int, val label: String) {
    MANUAL(0x00, "手动遥控"),
    VIDEO_CALL(0x01, "实时视频"),
    SMART_FOLLOW(0x02, "智能跟随"),
    PATROL(0x03, "智能巡逻"),
    VIDEO_OFF(0x04, "关闭视频");

    companion object {
        fun fromWire(value: Int): ControlMode =
            entries.firstOrNull { it.wireValue == value } ?: MANUAL
    }
}

data class GamepadState(
    val lx: Int = 128,
    val ly: Int = 128,
    val rx: Int = 128,
    val ry: Int = 128,
    val buttons: Int = 0
) {
    fun toPacket(): ByteArray {
        val f = CarSettings.maxSpeedPercent / 100.0
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        packet[0] = ControlProtocol.HEADER_0.toByte()
        packet[1] = ControlProtocol.HEADER_1.toByte()
        packet[2] = ControlProtocol.TYPE_GAMEPAD.toByte()
        packet[3] = scaleAxis(lx, f).toByte()
        packet[4] = scaleAxis(ly, f).toByte()
        packet[5] = scaleAxis(rx, f).toByte()
        packet[6] = scaleAxis(ry, f).toByte()
        packet[7] = (buttons and 0xFF).toByte()
        packet[8] = ((buttons ushr 8) and 0xFF).toByte()
        packet[9] = ControlProtocol.checksum(packet)
        return packet
    }

    fun toHexLine(): String = toPacket().toHexLine()
}

/**
 * 全局小车设置：最大速度百分比(10~100)，默认 30%。
 * demo/发送端/接收端 只要经 GamepadState.toPacket() 发指令都会受此限制。
 */
object CarSettings {
    @Volatile var maxSpeedPercent: Int = 30
}

private fun scaleAxis(axis: Int, f: Double): Int {
    val v = (axis - 128) * f + 128
    return v.roundToInt().coerceIn(0, 255)
}

object ModeCommand {
    fun packet(mode: ControlMode): ByteArray {
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        packet[0] = ControlProtocol.HEADER_0.toByte()
        packet[1] = ControlProtocol.HEADER_1.toByte()
        packet[2] = ControlProtocol.TYPE_MODE.toByte()
        packet[3] = mode.wireValue.toByte()
        packet[9] = ControlProtocol.checksum(packet)
        return packet
    }
}

object ServoCommand {
    fun packet(index: Int = 0, angle: Int): ByteArray {
        val angleClamped = angle.coerceIn(0, 180)
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        packet[0] = ControlProtocol.HEADER_0.toByte()
        packet[1] = ControlProtocol.HEADER_1.toByte()
        packet[2] = ControlProtocol.TYPE_SERVO.toByte()
        packet[3] = (index and 0xFF).toByte()
        packet[4] = angleClamped.toByte()
        packet[9] = ControlProtocol.checksum(packet)
        return packet
    }
}

/** 接收端 → 发送端 的状态回报: [3]=ESP32是否已连接(0/1), [4]=接收端当前模式 */
object StatusCommand {
    fun packet(espConnected: Boolean, mode: ControlMode): ByteArray {
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        packet[0] = ControlProtocol.HEADER_0.toByte()
        packet[1] = ControlProtocol.HEADER_1.toByte()
        packet[2] = ControlProtocol.TYPE_STATUS.toByte()
        packet[3] = (if (espConnected) 1 else 0).toByte()
        packet[4] = mode.wireValue.toByte()
        packet[9] = ControlProtocol.checksum(packet)
        return packet
    }
}

/** 发送端 → 接收端 的指令: [3]=命令码(CMD_*) */
object ControlCommand {
    fun packet(code: Int): ByteArray {
        val packet = ByteArray(ControlProtocol.PACKET_SIZE)
        packet[0] = ControlProtocol.HEADER_0.toByte()
        packet[1] = ControlProtocol.HEADER_1.toByte()
        packet[2] = ControlProtocol.TYPE_CMD.toByte()
        packet[3] = (code and 0xFF).toByte()
        packet[9] = ControlProtocol.checksum(packet)
        return packet
    }
}

object ControlProtocol {
    const val PACKET_SIZE = 10
    const val HEADER_0 = 0xAA
    const val HEADER_1 = 0x55
    const val TYPE_GAMEPAD = 0x01
    const val TYPE_MODE = 0x02
    const val TYPE_SERVO = 0x03
    const val TYPE_STATUS = 0x04
    const val TYPE_CMD = 0x05
    const val TYPE_DEBUG_ACK = 0x06
    const val CMD_CONNECT_ESP32 = 0x01

    fun parse(packet: ByteArray): ControlFrame? {
        if (packet.size != PACKET_SIZE) return null
        if ((packet[0].toInt() and 0xFF) != HEADER_0) return null
        if ((packet[1].toInt() and 0xFF) != HEADER_1) return null
        if (checksum(packet) != packet[9]) return null

        return when (packet[2].toInt() and 0xFF) {
            TYPE_GAMEPAD -> ControlFrame.Gamepad(
                GamepadState(
                    lx = packet[3].toInt() and 0xFF,
                    ly = packet[4].toInt() and 0xFF,
                    rx = packet[5].toInt() and 0xFF,
                    ry = packet[6].toInt() and 0xFF,
                    buttons = (packet[7].toInt() and 0xFF) or ((packet[8].toInt() and 0xFF) shl 8)
                )
            )
            TYPE_MODE -> ControlFrame.Mode(ControlMode.fromWire(packet[3].toInt() and 0xFF))
            TYPE_SERVO -> ControlFrame.Servo(
                index = packet[3].toInt() and 0xFF,
                angle = (packet[4].toInt() and 0xFF).coerceIn(0, 180)
            )
            TYPE_STATUS -> ControlFrame.Status(
                espConnected = (packet[3].toInt() and 0xFF) != 0,
                mode = ControlMode.fromWire(packet[4].toInt() and 0xFF)
            )
            TYPE_CMD -> ControlFrame.Command(packet[3].toInt() and 0xFF)
            TYPE_DEBUG_ACK -> ControlFrame.DebugAck(
                seq = (packet[3].toInt() and 0xFF) or
                    ((packet[4].toInt() and 0xFF) shl 8) or
                    ((packet[5].toInt() and 0xFF) shl 16) or
                    ((packet[6].toInt() and 0xFF) shl 24)
            )
            else -> null
        }
    }

    fun checksum(packet: ByteArray): Byte {
        var sum = 0
        for (i in 0 until PACKET_SIZE - 1) {
            sum = (sum + (packet[i].toInt() and 0xFF)) and 0xFF
        }
        return sum.toByte()
    }
}

object GamepadButtons {
    const val A = 1 shl 0
    const val B = 1 shl 1
    const val X = 1 shl 3
    const val Y = 1 shl 4
    const val L1 = 1 shl 6
    const val R1 = 1 shl 7
    const val L2 = 1 shl 8
    const val R2 = 1 shl 9
}

fun ByteArray.toHexLine(): String = joinToString(" ") { "%02X".format(it.toInt() and 0xFF) }
