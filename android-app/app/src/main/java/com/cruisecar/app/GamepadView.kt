package com.cruisecar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

class GamepadView(context: Context) : View(context) {
    var onStateChanged: ((GamepadState) -> Unit)? = null

    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(37, 43, 52) }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(117, 133, 154)
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val activePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(58, 151, 212) }
    private val knobPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(230, 236, 243) }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        textSize = 34f
    }

    private var leftPointer = -1
    private var leftDx = 0f
    private var leftDy = 0f

    override fun onDraw(canvas: Canvas) {
        drawStick(canvas, leftCenterX(), stickCenterY(), stickRadius(), leftDx, leftDy, "Lx/Ly")
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> handleDown(event.actionIndex, event)
            MotionEvent.ACTION_MOVE -> {
                updateSticks(event)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> handleUp(event.actionIndex, event)
            MotionEvent.ACTION_CANCEL -> resetControls()
        }
        invalidate()
        emitState()
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun handleDown(index: Int, event: MotionEvent) {
        val id = event.getPointerId(index)
        val x = event.getX(index)
        val y = event.getY(index)
        when {
            isInsideStick(x, y, leftCenterX(), stickCenterY()) && leftPointer == -1 -> leftPointer = id
        }
        updateSticks(event)
    }

    private fun handleUp(index: Int, event: MotionEvent) {
        val id = event.getPointerId(index)
        if (event.actionMasked == MotionEvent.ACTION_UP) {
            performClick()
        }
        if (id == leftPointer) {
            leftPointer = -1
            leftDx = 0f
            leftDy = 0f
        }
    }

    private fun resetControls() {
        leftPointer = -1
        leftDx = 0f
        leftDy = 0f
    }

    private fun updateSticks(event: MotionEvent) {
        updateStick(event, leftPointer, leftCenterX(), stickCenterY()) { dx, dy ->
            leftDx = dx
            leftDy = dy
        }
    }

    private fun updateStick(
        event: MotionEvent,
        pointerId: Int,
        centerX: Float,
        centerY: Float,
        update: (Float, Float) -> Unit
    ) {
        if (pointerId == -1) return
        val index = event.findPointerIndex(pointerId)
        if (index < 0) return
        val rawDx = event.getX(index) - centerX
        val rawDy = event.getY(index) - centerY
        val distance = hypot(rawDx, rawDy)
        val limit = stickRadius()
        if (distance <= limit || distance == 0f) {
            update(rawDx, rawDy)
        } else {
            val angle = atan2(rawDy, rawDx)
            update(cos(angle) * limit, sin(angle) * limit)
        }
    }

    private fun drawStick(canvas: Canvas, cx: Float, cy: Float, radius: Float, dx: Float, dy: Float, label: String) {
        canvas.drawCircle(cx, cy, radius, panelPaint)
        canvas.drawCircle(cx, cy, radius, strokePaint)
        canvas.drawCircle(cx + dx, cy + dy, radius * 0.36f, knobPaint)
        canvas.drawText(label, cx, cy + radius + 42f, textPaint)
    }

    private fun emitState() {
        val radius = stickRadius().coerceAtLeast(1f)
        onStateChanged?.invoke(
            GamepadState(
                lx = axisFromDelta(leftDx, radius),
                ly = axisFromDelta(leftDy, radius),
                rx = 128,
                ry = 128,
                buttons = 0
            )
        )
    }

    private fun axisFromDelta(delta: Float, radius: Float): Int =
        (128 + (delta / radius * 127f)).roundToInt().coerceIn(0, 255)

    private fun isInsideStick(x: Float, y: Float, cx: Float, cy: Float): Boolean =
        hypot(x - cx, y - cy) <= stickRadius() * 1.35f

    private fun stickRadius(): Float = min(width, height) * 0.11f
    private fun stickCenterY(): Float = height * 0.74f
    private fun leftCenterX(): Float = width * 0.22f
}
