package com.cruisecar.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean

data class ObjectDetection(
    val label: String,
    val confidence: Float,
    val rect: RectF
)

class ObjectRecognitionDemoController(
    private val frameProvider: () -> Bitmap?,
    private val onDetections: (List<ObjectDetection>) -> Unit
) {
    private val running = AtomicBoolean(false)

    fun start(onLog: (String) -> Unit) {
        if (!OpenCVLoader.initDebug()) {
            onLog("OpenCV init failed")
            return
        }
        if (!running.compareAndSet(false, true)) return

        Thread {
            onLog("OpenCV object demo started")
            while (running.get()) {
                val detections = frameProvider()?.let { analyze(it) }.orEmpty()
                onDetections(detections)
                Thread.sleep(220)
            }
        }.start()
    }

    fun stop() {
        running.set(false)
        onDetections(emptyList())
    }

    private fun analyze(bitmap: Bitmap): List<ObjectDetection> {
        val rgba = Mat()
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        val mergedMask = Mat()
        val hierarchy = Mat()
        val contours = mutableListOf<MatOfPoint>()

        return try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

            val detections = mutableListOf<ObjectDetection>()
            for (target in TARGETS) {
                buildMask(hsv, target, mask, mergedMask)
                Imgproc.erode(mergedMask, mergedMask, Mat())
                Imgproc.dilate(mergedMask, mergedMask, Mat())
                Imgproc.findContours(mergedMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

                for (contour in contours) {
                    val area = Imgproc.contourArea(contour)
                    val frameArea = (rgba.width() * rgba.height()).coerceAtLeast(1).toDouble()
                    val areaRatio = area / frameArea
                    if (areaRatio < MIN_AREA_RATIO) continue

                    val rect = Imgproc.boundingRect(contour)
                    val shape = classifyShape(contour, rect)
                    detections.add(
                        ObjectDetection(
                            label = "${target.label} $shape",
                            confidence = areaRatio.coerceIn(0.0, 1.0).toFloat(),
                            rect = rect.normalized(rgba.width(), rgba.height())
                        )
                    )
                }
                contours.forEach { it.release() }
                contours.clear()
            }

            detections
                .sortedByDescending { it.confidence }
                .take(MAX_DETECTIONS)
        } finally {
            rgba.release()
            rgb.release()
            hsv.release()
            mask.release()
            mergedMask.release()
            hierarchy.release()
            contours.forEach { it.release() }
        }
    }

    private fun buildMask(hsv: Mat, target: ColorTarget, mask: Mat, mergedMask: Mat) {
        Core.inRange(hsv, target.ranges.first().lower, target.ranges.first().upper, mergedMask)
        for (range in target.ranges.drop(1)) {
            Core.inRange(hsv, range.lower, range.upper, mask)
            Core.bitwise_or(mergedMask, mask, mergedMask)
        }
    }

    private fun classifyShape(contour: MatOfPoint, rect: Rect): String {
        val contour2f = MatOfPoint2f(*contour.toArray())
        val approx = MatOfPoint2f()
        return try {
            val perimeter = Imgproc.arcLength(contour2f, true)
            Imgproc.approxPolyDP(contour2f, approx, 0.04 * perimeter, true)
            val vertices = approx.total().toInt()
            val aspect = rect.width.toFloat() / rect.height.coerceAtLeast(1)
            when {
                vertices in 4..5 && aspect in 0.75f..1.33f -> "rectangle"
                vertices > 7 -> "circle"
                else -> "object"
            }
        } finally {
            contour2f.release()
            approx.release()
        }
    }

    private fun Rect.normalized(width: Int, height: Int): RectF =
        RectF(
            x.toFloat() / width.coerceAtLeast(1),
            y.toFloat() / height.coerceAtLeast(1),
            (x + this.width).toFloat() / width.coerceAtLeast(1),
            (y + this.height).toFloat() / height.coerceAtLeast(1)
        )

    private data class HsvRange(val lower: Scalar, val upper: Scalar)

    private data class ColorTarget(val label: String, val ranges: List<HsvRange>)

    companion object {
        private const val MIN_AREA_RATIO = 0.012
        private const val MAX_DETECTIONS = 6

        private val TARGETS = listOf(
            ColorTarget(
                "red",
                listOf(
                    HsvRange(Scalar(0.0, 70.0, 50.0), Scalar(10.0, 255.0, 255.0)),
                    HsvRange(Scalar(170.0, 70.0, 50.0), Scalar(180.0, 255.0, 255.0))
                )
            ),
            ColorTarget("green", listOf(HsvRange(Scalar(35.0, 55.0, 45.0), Scalar(90.0, 255.0, 255.0)))),
            ColorTarget("blue", listOf(HsvRange(Scalar(95.0, 55.0, 45.0), Scalar(130.0, 255.0, 255.0)))),
            ColorTarget("yellow", listOf(HsvRange(Scalar(18.0, 70.0, 60.0), Scalar(34.0, 255.0, 255.0))))
        )
    }
}

class ObjectRecognitionOverlayView(context: android.content.Context) : View(context) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 180, 140)
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(190, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
    }
    private var detections: List<ObjectDetection> = emptyList()

    fun setDetections(next: List<ObjectDetection>) {
        detections = next
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (detection in detections) {
            val rect = RectF(
                detection.rect.left * width,
                detection.rect.top * height,
                detection.rect.right * width,
                detection.rect.bottom * height
            )
            canvas.drawRect(rect, boxPaint)

            val label = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val textWidth = labelPaint.measureText(label)
            val bg = RectF(rect.left, (rect.top - 38f).coerceAtLeast(0f), rect.left + textWidth + 16f, rect.top)
            canvas.drawRect(bg, labelBgPaint)
            canvas.drawText(label, bg.left + 8f, bg.bottom - 9f, labelPaint)
        }
    }
}
