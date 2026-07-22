package com.cruisecar.app

data class GamepadState(
    val lx: Int = 128,
    val ly: Int = 128,
    val rx: Int = 128,
    val ry: Int = 128,
    val buttons: Int = 0
) {
    fun toPacket(): ByteArray {
        val packet = ByteArray(10)
        packet[0] = 0xAA.toByte()
        packet[1] = 0x55
        packet[2] = 0x01
        packet[3] = lx.coerceIn(0, 255).toByte()
        packet[4] = ly.coerceIn(0, 255).toByte()
        packet[5] = rx.coerceIn(0, 255).toByte()
        packet[6] = ry.coerceIn(0, 255).toByte()
        packet[7] = (buttons and 0xFF).toByte()
        packet[8] = ((buttons ushr 8) and 0xFF).toByte()
        packet[9] = checksum(packet)
        return packet
    }

    fun toHexLine(): String = toPacket().joinToString(" ") { "%02X".format(it.toInt() and 0xFF) }

    companion object {
        fun checksum(packet: ByteArray): Byte {
            var sum = 0
            for (i in 0 until 9) {
                sum = (sum + (packet[i].toInt() and 0xFF)) and 0xFF
            }
            return sum.toByte()
        }
    }
}

