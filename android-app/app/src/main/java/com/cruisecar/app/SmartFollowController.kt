package com.cruisecar.app

import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean

class SmartFollowController(
    private val frameProvider: () -> Bitmap?,
    private val onState: (GamepadState) -> Unit
) {
    private val running = AtomicBoolean(false)

    fun start(onLog: (String) -> Unit) {
        if (!OpenCVLoader.initDebug()) {
            onLog("OpenCV init failed")
            return
        }
        if (!running.compareAndSet(false, true)) return
        Thread {
            onLog("Smart follow started")
            while (running.get()) {
                val state = frameProvider()?.let { analyze(it) } ?: GamepadState()
                onState(state)
                Thread.sleep(160)
            }
        }.start()
    }

    fun stop() {
        running.set(false)
        onState(GamepadState())
    }

    private fun analyze(bitmap: Bitmap): GamepadState {
        val rgba = Mat()
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        val contours = mutableListOf<MatOfPoint>()
        return try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

            Core.inRange(hsv, Scalar(35.0, 60.0, 50.0), Scalar(90.0, 255.0, 255.0), mask)
            Imgproc.erode(mask, mask, Mat())
            Imgproc.dilate(mask, mask, Mat())
            Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val best = contours.maxByOrNull { Imgproc.contourArea(it) }
            if (best == null) {
                GamepadState()
            } else {
                val area = Imgproc.contourArea(best)
                val moments = Imgproc.moments(best)
                val center = if (moments.m00 != 0.0) Point(moments.m10 / moments.m00, moments.m01 / moments.m00) else Point(rgba.width() / 2.0, rgba.height() / 2.0)
                val errorX = ((center.x / rgba.width()) - 0.5).coerceIn(-0.5, 0.5)
                val areaRatio = area / (rgba.width() * rgba.height()).coerceAtLeast(1)
                val steering = (128 + errorX * 170).toInt().coerceIn(0, 255)
                val throttle = when {
                    areaRatio < 0.035 -> 92
                    areaRatio > 0.16 -> 164
                    else -> 128
                }
                GamepadState(lx = steering, ly = throttle)
            }
        } finally {
            rgba.release()
            rgb.release()
            hsv.release()
            mask.release()
            contours.forEach { it.release() }
        }
    }
}
