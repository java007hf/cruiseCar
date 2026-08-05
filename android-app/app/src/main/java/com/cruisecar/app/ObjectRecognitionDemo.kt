package com.cruisecar.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val MODEL_ASSET_NAME = "detect.tflite"
private const val LABELS_ASSET_NAME = "labels.txt"
// NOTE: 0.35 → 0.20 for the small orange-can single-class dataset trained
// on only ~50 near-frontal frames.  At 0.35, a 15° camera pitch change drops
// raw tflite confidence from 0.79 to ~0.28 (below threshold → 0 detections).
// Lowering to 0.20 tolerates the extra viewpoint variance until we re-train
// with a dataset that covers more angles / distances / backgrounds.  Revisit
// this threshold once the training dataset grows beyond ~200 diverse frames.
private const val CONFIDENCE_THRESHOLD = 0.20f
private const val IOU_THRESHOLD = 0.45f
private const val MAX_DETECTIONS = 5
private const val LOOP_INTERVAL_MS = 30L

private enum class ModelInputLayout {
    NHWC,
    NCHW
}

data class ObjectDetection(
    val label: String,
    val confidence: Float,
    val rect: RectF
)

class ObjectRecognitionDemoController(
    private val context: Context,
    private val frameProvider: () -> Bitmap?,
    private val onDetections: (List<ObjectDetection>) -> Unit
) {
    private val running = AtomicBoolean(false)
    private var detector: YoloTfliteDetector? = null
    private var workerThread: Thread? = null

    fun start(onLog: (String) -> Unit) {
        if (!running.compareAndSet(false, true)) return

        val activeDetector = YoloTfliteDetector.create(context, onLog)
        if (activeDetector == null) {
            running.set(false)
            return
        }
        detector = activeDetector

        workerThread = Thread {
            onLog("YOLO TFLite object detector started")
            try {
                while (running.get()) {
                    val detections = frameProvider()?.let { activeDetector.detect(it) }.orEmpty()
                    if (running.get()) onDetections(detections)
                    Thread.sleep(LOOP_INTERVAL_MS)
                }
            } catch (_: InterruptedException) {
            } finally {
                activeDetector.close()
                if (detector === activeDetector) detector = null
            }
        }.apply {
            name = "ObjectRecognitionDetector"
            start()
        }
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        workerThread?.interrupt()
        if (Thread.currentThread() !== workerThread) {
            runCatching { workerThread?.join(500) }
        }
        workerThread = null
        onDetections(emptyList())
    }
}

private class YoloTfliteDetector(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val inputLayout: ModelInputLayout,
    private val inputType: DataType,
    /** Debug logger (same sink as Controller.start()).  Emits per-frame
     *  top-confidence diagnostics so we can quickly tell the difference
     *  between "the model gives 0.9 but threshold filters it out" vs
     *  "the model genuinely outputs 0.00x noise". */
    private val onLog: (String) -> Unit,
) {
    fun detect(bitmap: Bitmap): List<ObjectDetection> {
        // Letterbox (preserves aspect ratio with grey padding) instead of the
        // previous stretch-to-square (Bitmap.createScaledBitmap).  The PyTorch
        // / ONNX training / export pipeline uses letterbox (see ultralytics
        // LetterBox / val transforms).  Switching this on Android is worth a
        // ~1.5x jump in raw ONNX confidence on the reference frame (see
        // _debug_pt_vs_onnx.py).  Without this fix, even a correct TFLite
        // conversion would lose detection recall on real camera frames that
        // don't match the 1:1 model input aspect.
        val (letterboxed, letterboxMeta) = letterbox(bitmap, inputWidth, inputHeight)
        val input = bitmapToInputBuffer(letterboxed)
        if (letterboxed !== bitmap) letterboxed.recycle()

        val outputShape = interpreter.getOutputTensor(0).shape()
        if (outputShape.size != 3 || outputShape[0] != 1) return emptyList()

        val output = Array(1) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
        interpreter.run(input, output)

        // ---- diagnostics: global max conf across ALL 8400 candidates
        // before any threshold / NMS.  Lets us tell "threshold too high"
        // (e.g. max conf=0.28, threshold=0.35 → 0 detections) from
        // "model genuinely can't see it" (max conf=0.00x).  Cheap to
        // compute (<1ms on 8400 boxes).
        val (rawMaxConf, rawAboveThreshold) = scanRawMaxConf(
            output[0], outputShape[1], outputShape[2]
        )

        val detections = parseYoloOutput(output[0], outputShape[1], outputShape[2])
        if (rawMaxConf < 0.01f || detections.isEmpty()) {
            // Throttle: only log every ~8th frame at 30ms loop (~3.75 logs/s).
            // Full blast would produce ~1 MB/s of noise in logcat.
            if (sLogThrottle++ and 7 == 0) {
                onLog(
                    "YOLO frame ${bitmap.width}x${bitmap.height}: " +
                        "raw_top_conf=%.4f raw_ge_thresh(%.2f)=%d final_det=%d".format(
                            rawMaxConf, CONFIDENCE_THRESHOLD, rawAboveThreshold, detections.size
                        )
                )
            }
            return detections
        }
        // Non-empty detections: always log the top result.
        val best = detections.first()
        onLog(
            "YOLO +hit '%s' conf=%.4f (raw_top=%.4f, raw_ge_thresh=%d) rect=[%.3f,%.3f,%.3f,%.3f]".format(
                best.label, best.confidence, rawMaxConf, rawAboveThreshold,
                best.rect.left, best.rect.top, best.rect.right, best.rect.bottom
            )
        )
        // Reverse the letterbox: model-space normalized rects come back in the
        // padded 640x640 coordinate frame.  Translate them back into the
        // original input `bitmap` so callers (OverlayView.drawDetections) can
        // continue to multiply by the overlay View's width/height.  This
        // preserves the detect() contract of returning rects normalized to
        // the *input* bitmap, not the letterboxed one.
        val (scale, dh, dw) = letterboxMeta
        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()
        val inW = inputWidth.toFloat()
        val inH = inputHeight.toFloat()
        return detections.map { det ->
            val r = det.rect
            // letterbox-norm → letterbox-px
            val lPx = r.left * inW
            val tPx = r.top * inH
            val rPx = r.right * inW
            val bPx = r.bottom * inH
            // remove letterbox padding → scaled resize px
            val lRs = (lPx - dw) / scale
            val tRs = (tPx - dh) / scale
            val rRs = (rPx - dw) / scale
            val bRs = (bPx - dh) / scale
            // resize px → original src-norm
            val left = (lRs / srcW).coerceIn(0f, 1f)
            val top = (tRs / srcH).coerceIn(0f, 1f)
            val right = (rRs / srcW).coerceIn(0f, 1f)
            val bottom = (bRs / srcH).coerceIn(0f, 1f)
            if (right <= left || bottom <= top) return@map null
            det.copy(rect = RectF(left, top, right, bottom))
        }.filterNotNull()
    }

    /**
     * Resize `src` preserving aspect ratio, then pad the result with mid-grey
     * (114, 114, 114) so the output has exactly `targetW x targetH` pixels.
     * This matches ultralytics' default LetterBox preprocessing used during
     * train/val and the ONNX/TFLite export tests.
     *
     * Returns: the padded bitmap + (scale, dh, dw) metadata needed to map a
     * detection rect back from padded-space to original-space.
     *   scale  : src_px → resized_px multiplier (0 < scale ≤ 1)
     *   dh, dw : top / left padding (px) inside the returned padded bitmap
     */
    private fun letterbox(
        src: Bitmap,
        targetW: Int,
        targetH: Int,
        padColor: Int = Color.rgb(114, 114, 114)
    ): Pair<Bitmap, Triple<Float, Int, Int>> {
        val scale = minOf(targetW.toFloat() / src.width, targetH.toFloat() / src.height)
        val newW = (src.width * scale).roundToInt()
        val newH = (src.height * scale).roundToInt()
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(padColor)
        val dw = (targetW - newW) / 2
        val dh = (targetH - newH) / 2
        canvas.drawBitmap(resized, dw.toFloat(), dh.toFloat(), null)
        if (resized !== src) resized.recycle()
        return out to Triple(scale, dh, dw)
    }

    fun close() {
        interpreter.close()
    }

    /** Cheap full-scan of the raw YOLO output tensor BEFORE decodePrediction's
     *  CONFIDENCE_THRESHOLD filter / NMS.  Returns (max_conf_global, count_ge_threshold).
     *
     *  max_conf_global = over all 8400 anchors: max(objectness × best_class_score).
     *  count_ge_threshold = how many boxes have (objectness × best_class_score)
     *                       >= CONFIDENCE_THRESHOLD (pre-NMS / pre-letterbox reverse).
     *
     *  This is the single most useful diagnostic number for "偏一点就不能识别"
     *  style complaints:
     *   - max_conf_global=0.80 but count_ge_threshold=0 → threshold is too high
     *   - max_conf_global=0.02 → model genuinely doesn't see the object
     *     (different angle than training, background shift, lighting, etc.)
     *   - count_ge_threshold > 10 but final_det=0 after NMS → IoU or letterbox
     *     coordinate bug (would normally be caught by the Kotlin e2e test).
     */
    private fun scanRawMaxConf(
        output: Array<FloatArray>, firstDim: Int, secondDim: Int
    ): Pair<Float, Int> {
        val attrsFirst = firstDim < secondDim
        val boxCount = if (attrsFirst) secondDim else firstDim
        val attrCount = if (attrsFirst) firstDim else secondDim
        if (attrCount < 5) return 0f to 0
        val classOffset = if (attrCount > labels.size + 4) 5 else 4

        var maxConf = 0f
        var count = 0
        for (boxIndex in 0 until boxCount) {
            // Read attrs inline (avoids FloatArray allocation per box — 8400
            // allocations per frame would be ~400 KB/s of garbage even on
            // throttled logging path).
            val objectnessRaw = if (classOffset == 5) {
                if (attrsFirst) output[4][boxIndex] else output[boxIndex][4]
            } else 1f
            val objectness = objectnessRaw.coerceIn(0f, 1f)
            var bestClassScore = 0f
            for (i in classOffset until attrCount) {
                val s = if (attrsFirst) output[i][boxIndex] else output[boxIndex][i]
                if (s > bestClassScore) bestClassScore = s
            }
            val conf = (objectness * bestClassScore).coerceIn(0f, 1f)
            if (conf > maxConf) maxConf = conf
            if (conf >= CONFIDENCE_THRESHOLD) count++
        }
        return maxConf to count
    }

    // Per-detector-instance counter for throttling the no-detection debug log.
    // A companion object Int is shared across instances, which is fine for a
    // single-instance controller but semantically clearer to keep inside the
    // class as one counter per detector; Kotlin doesn't have Java-style
    // instance-scope 'static so we just put it next to other private vals.
    private var sLogThrottle: Int = 0

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val bytesPerChannel = if (inputType == DataType.FLOAT32) 4 else 1
        val buffer = ByteBuffer
            .allocateDirect(1 * inputWidth * inputHeight * 3 * bytesPerChannel)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (inputLayout == ModelInputLayout.NHWC) {
            for (pixel in pixels) {
                buffer.putChannel((pixel shr 16) and 0xFF)
                buffer.putChannel((pixel shr 8) and 0xFF)
                buffer.putChannel(pixel and 0xFF)
            }
        } else {
            for (pixel in pixels) buffer.putChannel((pixel shr 16) and 0xFF)
            for (pixel in pixels) buffer.putChannel((pixel shr 8) and 0xFF)
            for (pixel in pixels) buffer.putChannel(pixel and 0xFF)
        }
        buffer.rewind()
        return buffer
    }

    private fun ByteBuffer.putChannel(value: Int) {
        if (inputType == DataType.FLOAT32) {
            putFloat(value / 255f)
        } else {
            put(value.toByte())
        }
    }

    private fun parseYoloOutput(output: Array<FloatArray>, firstDim: Int, secondDim: Int): List<ObjectDetection> {
        val attrsFirst = firstDim < secondDim
        val boxCount = if (attrsFirst) secondDim else firstDim
        val attrCount = if (attrsFirst) firstDim else secondDim
        if (attrCount < 5) return emptyList()

        val detections = ArrayList<ObjectDetection>()
        for (boxIndex in 0 until boxCount) {
            val attrs = FloatArray(attrCount) { attrIndex ->
                if (attrsFirst) output[attrIndex][boxIndex] else output[boxIndex][attrIndex]
            }
            decodePrediction(attrs)?.let { detections += it }
        }
        return detections
            .sortedByDescending { it.confidence }
            .nms(IOU_THRESHOLD)
            .take(MAX_DETECTIONS)
    }

    private fun decodePrediction(attrs: FloatArray): ObjectDetection? {
        val cx = attrs[0]
        val cy = attrs[1]
        val w = attrs[2]
        val h = attrs[3]

        val classOffset = if (attrs.size > labels.size + 4) 5 else 4
        val objectness = if (classOffset == 5) attrs[4].coerceIn(0f, 1f) else 1f
        var bestClass = 0
        var bestClassScore = 0f
        for (i in classOffset until attrs.size) {
            if (attrs[i] > bestClassScore) {
                bestClassScore = attrs[i]
                bestClass = i - classOffset
            }
        }

        val confidence = (objectness * bestClassScore).coerceIn(0f, 1f)
        if (confidence < CONFIDENCE_THRESHOLD) return null

        val normalized = max(max(cx, cy), max(w, h)) <= 2f
        val scaleX = if (normalized) 1f else inputWidth.toFloat()
        val scaleY = if (normalized) 1f else inputHeight.toFloat()
        val left = ((cx - w / 2f) / scaleX).coerceIn(0f, 1f)
        val top = ((cy - h / 2f) / scaleY).coerceIn(0f, 1f)
        val right = ((cx + w / 2f) / scaleX).coerceIn(0f, 1f)
        val bottom = ((cy + h / 2f) / scaleY).coerceIn(0f, 1f)
        if (right <= left || bottom <= top) return null

        val label = labels.getOrElse(bestClass) { "object_$bestClass" }
        return ObjectDetection(label, confidence, RectF(left, top, right, bottom))
    }

    companion object {
        fun create(context: Context, onLog: (String) -> Unit): YoloTfliteDetector? {
            val model = runCatching { context.assets.openFd(MODEL_ASSET_NAME).use { it.mapModel() } }
                .onFailure { onLog("Missing $MODEL_ASSET_NAME. Put trained YOLO TFLite model in app/src/main/assets/") }
                .getOrNull() ?: return null
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            val interpreter = Interpreter(model, options)
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            if (inputShape.size != 4 || inputShape[0] != 1) {
                onLog("Unsupported model input shape: ${inputShape.joinToString()}")
                interpreter.close()
                return null
            }
            val inputLayout: ModelInputLayout
            val height: Int
            val width: Int
            if (inputShape[3] == 3) {
                inputLayout = ModelInputLayout.NHWC
                height = inputShape[1]
                width = inputShape[2]
            } else if (inputShape[1] == 3) {
                inputLayout = ModelInputLayout.NCHW
                height = inputShape[2]
                width = inputShape[3]
            } else {
                onLog("Expected RGB input, got: ${inputShape.joinToString()}")
                interpreter.close()
                return null
            }
            if (width <= 0 || height <= 0) {
                onLog("Invalid model input shape: ${inputShape.joinToString()}")
                interpreter.close()
                return null
            }

            val labels = context.loadLabels()
            onLog("Loaded YOLO model ${width}x$height $inputLayout labels=${labels.size}: ${labels.joinToString(" / ")}")

            // Sanity-check: label list size should match the number of class scores
            // in the model output tensor. If they disagree, detections will still
            // happen (bestClass is still a valid integer), but the displayed label
            // text will be wrong for out-of-range class IDs — and the user will
            // silently see "object_N" placeholders instead of real names.
            val outShape = interpreter.getOutputTensor(0).shape()
            if (outShape.size == 3 && outShape[0] == 1) {
                val attrDim = minOf(outShape[1], outShape[2])
                val classOffset = if (attrDim > labels.size + 4) 5 else 4
                val modelClasses = attrDim - classOffset
                if (modelClasses != labels.size) {
                    onLog("WARN: model declares $modelClasses output classes but labels.txt has ${labels.size} entries. " +
                          "Mismatched labels/dataset. Display names will be wrong for out-of-range class IDs. " +
                          "Expected first line of labels.txt to match dataset.yaml names[0].")
                }
            }
            return YoloTfliteDetector(
                interpreter, labels, width, height, inputLayout, inputTensor.dataType(), onLog
            )
        }
    }
}

private fun android.content.res.AssetFileDescriptor.mapModel(): MappedByteBuffer =
    FileInputStream(fileDescriptor).channel.use { channel ->
        channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

private fun Context.loadLabels(): List<String> =
    runCatching {
        // Explicit UTF-8: Android JVM default is already UTF-8, but being explicit
        // documents intent + avoids any future charset-default surprises. The file
        // is produced by train_server.py post_train_export() (also UTF-8), so the
        // two ends are guaranteed byte-identical.
        assets.open(LABELS_ASSET_NAME).bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }.getOrDefault(
        listOf("object").also {
            android.util.Log.w("ObjectRecognitionDemo", "Missing or unreadable $LABELS_ASSET_NAME — falling back to label=['object']. " +
                "TFLite detection will still work but display name is generic.")
        }
    )

private fun List<ObjectDetection>.nms(iouThreshold: Float): List<ObjectDetection> {
    val selected = ArrayList<ObjectDetection>()
    for (candidate in this) {
        if (selected.none { it.rect.iou(candidate.rect) > iouThreshold }) {
            selected += candidate
        }
    }
    return selected
}

private fun RectF.iou(other: RectF): Float {
    val left = max(left, other.left)
    val top = max(top, other.top)
    val right = min(right, other.right)
    val bottom = min(bottom, other.bottom)
    val intersection = max(0f, right - left) * max(0f, bottom - top)
    val union = width() * height() + other.width() * other.height() - intersection
    return if (union <= 0f) 0f else intersection / union
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
