package com.cruisecar.app.mvi

import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity

data class AppState(
    val connectionMode: ConnectionMode = ConnectionMode.LAN,
    val remoteHost: String = "",
    val remoteUsername: String = "",
    val remoteToken: String = "",
    val remoteDeviceId: String = "",
    val remoteSenderId: String = "",
    val remoteManagerBaseUrl: String = "",
    val remotePreferredRole: String = "sender",
    val lastRemoteDeviceId: String = "",
    val lastRemoteDeviceName: String = "",
    val lastRemoteDeviceOnline: Boolean = false,
    val lastRemoteDeviceEspConnected: Boolean = false,
    val lastRemoteDeviceMode: String = "manual",
    val receiverIdentity: ReceiverIdentity? = null
)
