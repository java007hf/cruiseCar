package com.cruisecar.app.mvi

import com.cruisecar.app.data.local.ReceiverIdentityStore
import com.cruisecar.app.domain.model.ReceiverIdentity

class MainViewModel(
    private val receiverIdentityStore: ReceiverIdentityStore
) {
    var state: AppState = receiverIdentityStore.getRemoteAccount().let { account ->
        AppState(
            remoteHost = account.host.ifBlank { DEFAULT_REMOTE_HOST },
            remoteUsername = account.username,
            remoteToken = account.token,
            remoteDeviceId = account.lastDeviceId,
            remoteSenderId = account.senderId,
            remoteManagerBaseUrl = account.managerBaseUrl,
            remotePreferredRole = account.preferredRole.ifBlank { "sender" },
            lastRemoteDeviceId = account.lastDeviceId,
            lastRemoteDeviceName = account.lastDeviceName,
            lastRemoteDeviceOnline = account.lastDeviceOnline,
            lastRemoteDeviceEspConnected = account.lastDeviceEspConnected,
            lastRemoteDeviceMode = account.lastDeviceMode
        )
    }
        private set

    fun dispatch(intent: AppIntent) {
        state = when (intent) {
            is AppIntent.SetConnectionMode -> state.copy(connectionMode = intent.mode)
            is AppIntent.SetRemoteDeviceId -> state.copy(remoteDeviceId = intent.deviceId)
            is AppIntent.SetRemoteSenderId -> state.copy(remoteSenderId = intent.senderId)
            is AppIntent.SetRemotePreferredRole -> {
                receiverIdentityStore.savePreferredRemoteRole(intent.role)
                state.copy(remotePreferredRole = intent.role)
            }
            is AppIntent.SetLastRemoteDevice -> {
                receiverIdentityStore.saveLastRemoteDevice(
                    intent.deviceId,
                    intent.name,
                    intent.online,
                    intent.espConnected,
                    intent.mode
                )
                state.copy(
                    remoteDeviceId = intent.deviceId,
                    lastRemoteDeviceId = intent.deviceId,
                    lastRemoteDeviceName = intent.name,
                    lastRemoteDeviceOnline = intent.online,
                    lastRemoteDeviceEspConnected = intent.espConnected,
                    lastRemoteDeviceMode = intent.mode
                )
            }
            is AppIntent.ReceiverIdentityLoaded -> state.copy(receiverIdentity = intent.identity)
        }
    }

    fun receiverIdentity(): ReceiverIdentity {
        val identity = state.receiverIdentity ?: receiverIdentityStore.getOrCreate()
        if (state.receiverIdentity == null) {
            dispatch(AppIntent.ReceiverIdentityLoaded(identity))
        }
        return identity
    }

    fun saveRemoteAccount(host: String, username: String, token: String, managerBaseUrl: String) {
        val senderId = state.remoteSenderId.ifBlank { "phone-${System.currentTimeMillis() % 100000}" }
        receiverIdentityStore.saveRemoteAccount(host, username, token, managerBaseUrl, senderId)
        state = state.copy(
            remoteHost = host,
            remoteUsername = username,
            remoteToken = token,
            remoteManagerBaseUrl = managerBaseUrl,
            remoteSenderId = senderId
        )
    }

    private companion object {
        const val DEFAULT_REMOTE_HOST = "116.62.32.90"
    }
}
