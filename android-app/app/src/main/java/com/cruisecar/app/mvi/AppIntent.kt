package com.cruisecar.app.mvi

import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity

sealed class AppIntent {
    data class SetConnectionMode(val mode: ConnectionMode) : AppIntent()
    data class SetRemoteHost(val host: String) : AppIntent()
    data class SetRemoteToken(val token: String) : AppIntent()
    data class SetRemoteDeviceId(val deviceId: String) : AppIntent()
    data class SetRemoteSenderId(val senderId: String) : AppIntent()
    data class SetRemoteManagerBaseUrl(val baseUrl: String) : AppIntent()
    data class ReceiverIdentityLoaded(val identity: ReceiverIdentity) : AppIntent()
}
