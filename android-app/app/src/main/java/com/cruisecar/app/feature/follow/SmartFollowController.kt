package com.cruisecar.app.feature.follow

import android.content.Context
import com.cruisecar.app.protocol.GamepadState
import android.graphics.Bitmap
import com.cruisecar.app.feature.vision.ObjectDetection
import com.cruisecar.app.feature.vision.ObjectRecognitionDemoController

class SmartFollowController(
    context: Context,
    private val frameProvider: () -> Bitmap?,
    private val onState: (GamepadState) -> Unit,
    private val behavior: Behavior = Behavior.FOLLOW
) {
    enum class Behavior { FIND, FOLLOW }

    private var missedFrames = 0
    private var foundLogged = false
    private val detector = ObjectRecognitionDemoController(
        context = context.applicationContext,
        frameProvider = frameProvider,
        onDetections = ::handleDetections
    )
    private var onLog: (String) -> Unit = {}

    fun start(onLog: (String) -> Unit) {
        this.onLog = onLog
        onLog("YOLO cola ${behavior.name.lowercase()} started")
        detector.start(onLog)
    }

    fun stop() {
        detector.stop()
        onState(GamepadState())
    }

    private fun handleDetections(detections: List<ObjectDetection>) {
        val target = detections.maxByOrNull { it.confidence }
        if (target == null) {
            missedFrames++
            // Wait for a few frames to avoid steering on a single inference miss,
            // then rotate slowly until the bottle is visible again.
            onState(if (missedFrames >= 5) GamepadState(lx = 188, ly = 128) else GamepadState())
            foundLogged = false
            return
        }

        missedFrames = 0
        if (!foundLogged) {
            onLog("YOLO cola found confidence=${"%.2f".format(target.confidence)}")
            foundLogged = true
        }
        if (behavior == Behavior.FIND) {
            onState(GamepadState())
            return
        }

        val rect = target.rect
        val centerX = (rect.left + rect.right) / 2f
        val errorX = (centerX - 0.5f).coerceIn(-0.5f, 0.5f)
        val areaRatio = ((rect.right - rect.left) * (rect.bottom - rect.top)).coerceAtLeast(0f)
        val steering = (128 + errorX * 170).toInt().coerceIn(0, 255)
        val throttle = when {
            areaRatio < 0.035f -> 92
            areaRatio > 0.16f -> 164
            else -> 128
        }
        onState(GamepadState(lx = steering, ly = throttle))
    }
}
