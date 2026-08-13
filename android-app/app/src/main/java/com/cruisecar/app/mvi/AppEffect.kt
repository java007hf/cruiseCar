package com.cruisecar.app.mvi

sealed class AppEffect {
    data class Toast(val message: String) : AppEffect()
    data class Log(val message: String) : AppEffect()
}
