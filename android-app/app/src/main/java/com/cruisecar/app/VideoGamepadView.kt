package com.cruisecar.app

import android.content.Context
import android.view.View
import android.widget.FrameLayout

/**
 * 复用组件：把“视频/预览背景”与“摇杆遥控”叠放在同一个容器里。
 *
 * - 背景由调用方通过 [setBackground] 提供（如 CameraPreviewView、WebRTC SurfaceViewRenderer）。
 * - 摇杆 [GamepadView] 始终浮在上层，并通过 [onStateChanged] 输出 [GamepadState]。
 *
 * DebugActivity、MainActivity(发送端) 共享此组件，保证多端遥控交互一致、便于功能移植。
 */
class VideoGamepadView(context: Context) : FrameLayout(context) {
    var onStateChanged: ((GamepadState) -> Unit)? = null

    private val gamepad = GamepadView(context).apply {
        onStateChanged = { this@VideoGamepadView.onStateChanged?.invoke(it) }
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
    }
}
