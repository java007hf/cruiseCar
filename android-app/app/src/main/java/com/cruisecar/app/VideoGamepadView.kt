package com.cruisecar.app

import android.content.Context
import android.view.View
import android.widget.FrameLayout

/**
 * 复用组件：把“视频/预览背景”与“摇杆遥控”“舵机上下控制”叠放在同一个容器里。
 *
 * - 背景由调用方通过 [setBackground] 提供（如 CameraPreviewView、WebRTC SurfaceViewRenderer）。
 * - 摇杆 [GamepadView] 浮在上层左侧，输出 [GamepadState]（[onStateChanged]）。
 * - 舵机竖向控制 [ServoVerticalView] 浮在上层右侧，输出舵机角度（[onServoChanged]）。
 *
 * DebugActivity、MainActivity(发送端) 共享此组件，保证多端遥控交互一致、便于功能移植，
 * “视频 + 控制 + 舵机”因此始终是同一个控件。
 */
class VideoGamepadView(context: Context) : FrameLayout(context) {
    var onStateChanged: ((GamepadState) -> Unit)? = null
    var onServoChanged: ((Int) -> Unit)? = null

    private val gamepad = GamepadView(context).apply {
        onStateChanged = { this@VideoGamepadView.onStateChanged?.invoke(it) }
    }
    private val servo = ServoVerticalView(context).apply {
        onServoChanged = { this@VideoGamepadView.onServoChanged?.invoke(it) }
    }
    private var backgroundView: View? = null

    init {
        addView(
            gamepad,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        addView(
            servo,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
    }

    /** 设置底层预览/视频视图（识别框、相机预览或 WebRTC 视频）。 */
    fun setBackground(view: View) {
        backgroundView?.let { removeView(it) }
        backgroundView = view
        addView(
            view,
            0,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        bringChildToFront(gamepad)
        bringChildToFront(servo)
    }
}
