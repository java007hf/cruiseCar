package com.cruisecar.app.mvi

import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity

sealed class AppIntent {
    data class SetConnectionMode(val mode: ConnectionMode) : AppIntent()
    data class SetRemoteDeviceId(val deviceId: String) : AppIntent()
    data class SetRemoteSenderId(val senderId: String) : AppIntent()
    data class SetRemotePreferredRole(val role: String) : AppIntent()
    data class SetLastRemoteDevice(
        val deviceId: String,
        val name: String,
        val online: Boolean,
        val espConnected: Boolean,
        val mode: String
    ) : AppIntent()
    data class ReceiverIdentityLoaded(val identity: ReceiverIdentity) : AppIntent()
}
