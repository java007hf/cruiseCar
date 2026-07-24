package com.cruisecar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(22, 26, 32) }
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
    private var rightPointer = -1
    private var leftDx = 0f
    private var leftDy = 0f
    private var rightDx = 0f
    private var rightDy = 0f
    private var buttons = 0

    private val buttonRects = mutableMapOf<Int, RectF>()
    private val buttonLabels = mapOf(
        GamepadButtons.Y to "Y",
        GamepadButtons.X to "X",
        GamepadButtons.B to "B",
        GamepadButtons.A to "A",
        GamepadButtons.L1 to "L1",
        GamepadButtons.R1 to "R1",
        GamepadButtons.L2 to "L2",
        GamepadButtons.R2 to "R2"
    )

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(bgPaint.color)
        layoutButtons()
        drawStick(canvas, leftCenterX(), stickCenterY(), stickRadius(), leftDx, leftDy, "L")
        drawStick(canvas, rightCenterX(), stickCenterY(), stickRadius(), rightDx, rightDy, "R")
        drawButtons(canvas)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> handleDown(event.actionIndex, event)
            MotionEvent.ACTION_MOVE -> {
                updateSticks(event)
                updateButtons(event)
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
            isInsideStick(x, y, rightCenterX(), stickCenterY()) && rightPointer == -1 -> rightPointer = id
            else -> setButtonAt(x, y, true)
        }
        updateSticks(event)
        updateButtons(event)
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
        if (id == rightPointer) {
            rightPointer = -1
            rightDx = 0f
            rightDy = 0f
        }
        updateButtons(event, releasedPointerId = id)
    }

    private fun resetControls() {
        leftPointer = -1
        rightPointer = -1
        leftDx = 0f
        leftDy = 0f
        rightDx = 0f
        rightDy = 0f
        buttons = 0
    }

    private fun updateSticks(event: MotionEvent) {
        updateStick(event, leftPointer, leftCenterX(), stickCenterY()) { dx, dy ->
            leftDx = dx
            leftDy = dy
        }
        updateStick(event, rightPointer, rightCenterX(), stickCenterY()) { dx, dy ->
            rightDx = dx
            rightDy = dy
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

    private fun updateButtons(event: MotionEvent, releasedPointerId: Int = -1) {
        var nextButtons = 0
        for (i in 0 until event.pointerCount) {
            if (event.getPointerId(i) == releasedPointerId) continue
            val id = event.getPointerId(i)
            if (id == leftPointer || id == rightPointer) continue
            val x = event.getX(i)
            val y = event.getY(i)
            for ((bit, rect) in buttonRects) {
                if (rect.contains(x, y)) nextButtons = nextButtons or bit
            }
        }
        buttons = nextButtons
    }

    private fun setButtonAt(x: Float, y: Float, pressed: Boolean) {
        for ((bit, rect) in buttonRects) {
            if (rect.contains(x, y)) {
                buttons = if (pressed) buttons or bit else buttons and bit.inv()
            }
        }
    }

    private fun drawStick(canvas: Canvas, cx: Float, cy: Float, radius: Float, dx: Float, dy: Float, label: String) {
        canvas.drawCircle(cx, cy, radius, panelPaint)
        canvas.drawCircle(cx, cy, radius, strokePaint)
        canvas.drawCircle(cx + dx, cy + dy, radius * 0.36f, knobPaint)
        canvas.drawText(label, cx, cy + radius + 42f, textPaint)
    }

    private fun drawButtons(canvas: Canvas) {
        for ((bit, rect) in buttonRects) {
            canvas.drawRoundRect(rect, 14f, 14f, if ((buttons and bit) != 0) activePaint else panelPaint)
            canvas.drawRoundRect(rect, 14f, 14f, strokePaint)
            val y = rect.centerY() - (textPaint.descent() + textPaint.ascent()) / 2
            canvas.drawText(buttonLabels[bit] ?: "", rect.centerX(), y, textPaint)
        }
    }

    private fun layoutButtons() {
        buttonRects.clear()
        val size = min(width, height) * 0.115f
        val gap = size * 0.18f
        val actionCx = width * 0.74f
        val actionCy = height * 0.38f
        buttonRects[GamepadButtons.Y] = square(actionCx, actionCy - size - gap, size)
        buttonRects[GamepadButtons.X] = square(actionCx - size - gap, actionCy, size)
        buttonRects[GamepadButtons.B] = square(actionCx + size + gap, actionCy, size)
        buttonRects[GamepadButtons.A] = square(actionCx, actionCy + size + gap, size)

        val shoulderTop = height * 0.08f
        buttonRects[GamepadButtons.L1] = RectF(width * 0.08f, shoulderTop, width * 0.28f, shoulderTop + size)
        buttonRects[GamepadButtons.R1] = RectF(width * 0.72f, shoulderTop, width * 0.92f, shoulderTop + size)
        buttonRects[GamepadButtons.L2] = RectF(width * 0.08f, shoulderTop + size + gap, width * 0.28f, shoulderTop + size * 2 + gap)
        buttonRects[GamepadButtons.R2] = RectF(width * 0.72f, shoulderTop + size + gap, width * 0.92f, shoulderTop + size * 2 + gap)
    }

    private fun square(cx: Float, cy: Float, size: Float): RectF =
        RectF(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)

    private fun emitState() {
        val radius = stickRadius().coerceAtLeast(1f)
        onStateChanged?.invoke(
            GamepadState(
                lx = axisFromDelta(leftDx, radius),
                ly = axisFromDelta(leftDy, radius),
                rx = axisFromDelta(rightDx, radius),
                ry = axisFromDelta(rightDy, radius),
                buttons = buttons
            )
        )
    }

    private fun axisFromDelta(delta: Float, radius: Float): Int =
        (128 + (delta / radius * 127f)).roundToInt().coerceIn(0, 255)

    private fun isInsideStick(x: Float, y: Float, cx: Float, cy: Float): Boolean =
        hypot(x - cx, y - cy) <= stickRadius() * 1.35f

    private fun stickRadius(): Float = min(width, height) * 0.16f
    private fun stickCenterY(): Float = height * 0.68f
    private fun leftCenterX(): Float = width * 0.24f
    private fun rightCenterX(): Float = width * 0.56f
}
