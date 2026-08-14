package com.cruisecar.app.data.local

import android.content.Context
import com.cruisecar.app.domain.model.ReceiverIdentity
import com.cruisecar.app.utils.DeviceIdUtils
import java.util.UUID

class ReceiverIdentityStore(context: Context) {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("receiver_identity", Context.MODE_PRIVATE)
    private val accountPrefs = appContext.getSharedPreferences("remote_account", Context.MODE_PRIVATE)

    fun getOrCreate(): ReceiverIdentity {
        val savedId = prefs.getString(KEY_DEVICE_ID, "").orEmpty()
        val savedName = prefs.getString(KEY_DISPLAY_NAME, "").orEmpty()
        if (savedId.isNotBlank() && savedName.isNotBlank()) {
            return ReceiverIdentity(savedId, savedName)
        }

        val installId = prefs.getString(KEY_INSTALL_ID, "").orEmpty().ifBlank { UUID.randomUUID().toString() }
        val identity = DeviceIdUtils.buildReceiverIdentity(appContext, installId)
        prefs.edit()
            .putString(KEY_INSTALL_ID, installId)
            .putString(KEY_DEVICE_ID, identity.deviceId)
            .putString(KEY_DISPLAY_NAME, identity.displayName)
            .apply()
        return identity
    }

    fun getRemoteAccount(): RemoteAccount {
        clearLegacySavedPassword()
        return RemoteAccount(
            host = accountPrefs.getString(KEY_REMOTE_HOST, "").orEmpty(),
            username = accountPrefs.getString(KEY_REMOTE_USERNAME, "").orEmpty(),
            token = accountPrefs.getString(KEY_REMOTE_TOKEN, "").orEmpty(),
            managerBaseUrl = accountPrefs.getString(KEY_REMOTE_MANAGER_BASE_URL, "").orEmpty(),
            senderId = accountPrefs.getString(KEY_REMOTE_SENDER_ID, "").orEmpty(),
            preferredRole = accountPrefs.getString(KEY_REMOTE_PREFERRED_ROLE, "sender").orEmpty(),
            lastDeviceId = accountPrefs.getString(KEY_REMOTE_LAST_DEVICE_ID, "").orEmpty(),
            lastDeviceName = accountPrefs.getString(KEY_REMOTE_LAST_DEVICE_NAME, "").orEmpty(),
            lastDeviceOnline = accountPrefs.getBoolean(KEY_REMOTE_LAST_DEVICE_ONLINE, false),
            lastDeviceEspConnected = accountPrefs.getBoolean(KEY_REMOTE_LAST_DEVICE_ESP_CONNECTED, false),
            lastDeviceMode = accountPrefs.getString(KEY_REMOTE_LAST_DEVICE_MODE, "manual").orEmpty()
        )
    }

    fun saveRemoteAccount(
        host: String,
        username: String,
        token: String,
        managerBaseUrl: String,
        senderId: String
    ) {
        accountPrefs.edit()
            .putString(KEY_REMOTE_HOST, host)
            .putString(KEY_REMOTE_USERNAME, username)
            .putString(KEY_REMOTE_TOKEN, token)
            .putString(KEY_REMOTE_MANAGER_BASE_URL, managerBaseUrl)
            .putString(KEY_REMOTE_SENDER_ID, senderId)
            .remove(KEY_LEGACY_REMOTE_PASSWORD)
            .apply()
    }

    fun savePreferredRemoteRole(role: String) {
        accountPrefs.edit()
            .putString(KEY_REMOTE_PREFERRED_ROLE, role)
            .apply()
    }

    fun saveLastRemoteDevice(
        deviceId: String,
        name: String,
        online: Boolean,
        espConnected: Boolean,
        mode: String
    ) {
        accountPrefs.edit()
            .putString(KEY_REMOTE_LAST_DEVICE_ID, deviceId)
            .putString(KEY_REMOTE_LAST_DEVICE_NAME, name)
            .putBoolean(KEY_REMOTE_LAST_DEVICE_ONLINE, online)
            .putBoolean(KEY_REMOTE_LAST_DEVICE_ESP_CONNECTED, espConnected)
            .putString(KEY_REMOTE_LAST_DEVICE_MODE, mode)
            .apply()
    }

    fun clearLastRemoteDevice() {
        accountPrefs.edit()
            .remove(KEY_REMOTE_LAST_DEVICE_ID)
            .remove(KEY_REMOTE_LAST_DEVICE_NAME)
            .remove(KEY_REMOTE_LAST_DEVICE_ONLINE)
            .remove(KEY_REMOTE_LAST_DEVICE_ESP_CONNECTED)
            .remove(KEY_REMOTE_LAST_DEVICE_MODE)
            .apply()
    }

    private fun clearLegacySavedPassword() {
        if (accountPrefs.contains(KEY_LEGACY_REMOTE_PASSWORD)) {
            accountPrefs.edit().remove(KEY_LEGACY_REMOTE_PASSWORD).apply()
        }
    }

    private companion object {
        const val KEY_INSTALL_ID = "install_id"
        const val KEY_DEVICE_ID = "device_id"
        const val KEY_DISPLAY_NAME = "display_name"
        const val KEY_REMOTE_HOST = "remote_host"
        const val KEY_REMOTE_USERNAME = "remote_username"
        const val KEY_LEGACY_REMOTE_PASSWORD = "remote_password"
        const val KEY_REMOTE_TOKEN = "remote_token"
        const val KEY_REMOTE_MANAGER_BASE_URL = "remote_manager_base_url"
        const val KEY_REMOTE_SENDER_ID = "remote_sender_id"
        const val KEY_REMOTE_PREFERRED_ROLE = "remote_preferred_role"
        const val KEY_REMOTE_LAST_DEVICE_ID = "remote_last_device_id"
        const val KEY_REMOTE_LAST_DEVICE_NAME = "remote_last_device_name"
        const val KEY_REMOTE_LAST_DEVICE_ONLINE = "remote_last_device_online"
        const val KEY_REMOTE_LAST_DEVICE_ESP_CONNECTED = "remote_last_device_esp_connected"
        const val KEY_REMOTE_LAST_DEVICE_MODE = "remote_last_device_mode"
    }
}

data class RemoteAccount(
    val host: String,
    val username: String,
    val token: String,
    val managerBaseUrl: String,
    val senderId: String,
    val preferredRole: String,
    val lastDeviceId: String,
    val lastDeviceName: String,
    val lastDeviceOnline: Boolean,
    val lastDeviceEspConnected: Boolean,
    val lastDeviceMode: String
)
