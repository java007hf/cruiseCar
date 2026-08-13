package com.cruisecar.app.utils

import android.content.Context
import android.os.Build
import android.provider.Settings
import com.cruisecar.app.domain.model.ReceiverIdentity
import java.security.MessageDigest
import java.util.Locale

object DeviceIdUtils {
    fun buildReceiverIdentity(context: Context, installId: String): ReceiverIdentity {
        val manufacturer = Build.MANUFACTURER.safeDevicePart().ifBlank { "Android" }
        val model = Build.MODEL.safeDevicePart().ifBlank { "Phone" }
        val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID).orEmpty()
        val seed = listOf(
            context.packageName,
            androidId,
            installId,
            Build.BRAND,
            Build.DEVICE,
            Build.MODEL,
            Build.MANUFACTURER
        ).joinToString("|")
        val suffix = sha256(seed).take(8).uppercase(Locale.US)
        val displayName = "${Build.MANUFACTURER.readableDevicePart()} ${Build.MODEL.readableDevicePart()}".trim()
            .ifBlank { "Android 接收端" }
        val deviceId = "car-${manufacturer}-${model}-$suffix"
            .lowercase(Locale.US)
            .replace(Regex("-+"), "-")
            .trim('-')
        return ReceiverIdentity(deviceId = deviceId, displayName = displayName)
    }

    private fun sha256(value: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it.toInt() and 0xFF) }
    }

    private fun String.safeDevicePart(): String = trim()
        .lowercase(Locale.US)
        .replace(Regex("[^a-z0-9]+"), "-")
        .trim('-')
        .take(24)

    private fun String.readableDevicePart(): String = trim()
        .replace(Regex("\\s+"), " ")
        .take(32)
}
