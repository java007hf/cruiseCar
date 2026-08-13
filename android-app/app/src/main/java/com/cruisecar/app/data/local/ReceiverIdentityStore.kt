package com.cruisecar.app.data.local

import android.content.Context
import com.cruisecar.app.domain.model.ReceiverIdentity
import com.cruisecar.app.utils.DeviceIdUtils
import java.util.UUID

class ReceiverIdentityStore(context: Context) {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("receiver_identity", Context.MODE_PRIVATE)

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

    private companion object {
        const val KEY_INSTALL_ID = "install_id"
        const val KEY_DEVICE_ID = "device_id"
        const val KEY_DISPLAY_NAME = "display_name"
    }
}
