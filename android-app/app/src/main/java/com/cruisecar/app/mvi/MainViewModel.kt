package com.cruisecar.app.mvi

import com.cruisecar.app.data.local.ReceiverIdentityStore
import com.cruisecar.app.domain.model.ReceiverIdentity

class MainViewModel(
    private val receiverIdentityStore: ReceiverIdentityStore
) {
    var state: AppState = AppState()
        private set

    fun dispatch(intent: AppIntent) {
        state = when (intent) {
            is AppIntent.SetConnectionMode -> state.copy(connectionMode = intent.mode)
            is AppIntent.SetRemoteHost -> state.copy(remoteHost = intent.host)
            is AppIntent.SetRemoteToken -> state.copy(remoteToken = intent.token)
            is AppIntent.SetRemoteDeviceId -> state.copy(remoteDeviceId = intent.deviceId)
            is AppIntent.SetRemoteSenderId -> state.copy(remoteSenderId = intent.senderId)
            is AppIntent.SetRemoteManagerBaseUrl -> state.copy(remoteManagerBaseUrl = intent.baseUrl)
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
}
