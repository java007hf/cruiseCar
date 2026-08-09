package com.cruisecar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * 视频容器内嵌的“舵机上下”控制：竖向摇杆风格，仅响应纵向拖拽。
 *
 * 角度范围限制为 130°~180°（舵机物理行程）：轨道底=130°，轨道顶=180°，
 * 旋钮只能在这一段内移动，松手后保持当前角度。
 *
 * 命中区仅在右侧轨道附近；左侧区域回调 false 不消费事件，由下层 [GamepadView] 处理，
 * 因此移动摇杆与舵机控制互不干扰。视觉沿用 Lx/Ly 摇杆的配色与旋钮风格，只是改成上下单向。
 */
class ServoVerticalView(context: Context) : View(context) {
    var onServoChanged: ((Int) -> Unit)? = null

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(37, 43, 52)
        style = Paint.Style.STROKE
        strokeWidth = 14f
        strokeCap = Paint.Cap.ROUND
    }
    private val knobPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(230, 236, 243) }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        textSize = 30f
    }

    private var pointer = -1
    private var knobFrac = 0f  // 0 = 底部(130°), 1 = 顶部(180°)

    private fun trackLen(): Float = height * 0.40f
    private fun bottomY(): Float = height * 0.74f           // 130°
    private fun topY(): Float = bottomY() - trackLen()      // 180°
    private fun centerX(): Float = width * 0.78f
    private fun knobRadius(): Float = (width * 0.07f).coerceAtMost(height * 0.06f)

    override fun onDraw(canvas: Canvas) {
        val cx = centerX()
        val top = topY()
        val bottom = bottomY()
        // 轨道 (130°→180°)
        canvas.drawLine(cx, top, cx, bottom, trackPaint)
        // 端点提示
        canvas.drawText("180°", cx, top - 14f, textPaint)
        canvas.drawText("130°", cx, bottom + 38f, textPaint)
        // 旋钮
        val knobY = bottom - knobFrac * trackLen()
        canvas.drawCircle(cx, knobY, knobRadius(), knobPaint)
        // 当前角度
        canvas.drawText("${currentAngle()}°", cx, bottom + 78f, textPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                val idx = event.actionIndex
                val id = event.getPointerId(idx)
                if (pointer == -1 && inside(event.getX(idx), event.getY(idx))) {
                    pointer = id
                    parent?.requestDisallowInterceptTouchEvent(true)
                    update(event)
                } else {
                    return false
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (pointer != -1) {
                    parent?.requestDisallowInterceptTouchEvent(true)
                    update(event)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                if (event.getPointerId(event.actionIndex) == pointer) {
                    pointer = -1
                    parent?.requestDisallowInterceptTouchEvent(false)
                }
            }
            MotionEvent.ACTION_CANCEL -> {
                pointer = -1
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        }
        invalidate()
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun update(event: MotionEvent) {
        val idx = event.findPointerIndex(pointer)
        if (idx < 0) return
        val len = trackLen().coerceAtLeast(1f)
        knobFrac = ((bottomY() - event.getY(idx)) / len).coerceIn(0f, 1f)
        onServoChanged?.invoke(currentAngle())
    }

    private fun currentAngle(): Int = (130 + knobFrac * 50).roundToInt().coerceIn(130, 180)

    private fun inside(x: Float, y: Float): Boolean {
        val cx = centerX()
        val margin = knobRadius() * 3f
        return abs(x - cx) <= margin && y in (topY() - margin)..(bottomY() + margin)
    }
}
