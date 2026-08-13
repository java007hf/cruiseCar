package com.cruisecar.app.mvi

import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity

data class AppState(
    val connectionMode: ConnectionMode = ConnectionMode.LAN,
    val remoteHost: String = "",
    val remoteUsername: String = "",
    val remotePassword: String = "",
    val remoteToken: String = "",
    val remoteDeviceId: String = "",
    val remoteSenderId: String = "",
    val remoteManagerBaseUrl: String = "",
    val receiverIdentity: ReceiverIdentity? = null
)
