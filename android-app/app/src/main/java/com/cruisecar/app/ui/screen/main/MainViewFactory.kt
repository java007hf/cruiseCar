package com.cruisecar.app.ui.screen.main

import android.content.Context
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

data class ScreenWithLog(
    val root: View,
    val logView: TextView,
    val logScroll: ScrollView,
    val olderLogsButton: Button,
    val latestLogsButton: Button
)

class MainViewFactory(private val context: Context) {
    fun rootLayout(): LinearLayout = LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(32, 32, 32, 32)
    }

    /** 返回一个可滚动根: 外层 ScrollView + 内层纵向 LinearLayout(用于 addView)。
     *  在 ScrollView 中 weight 不再分配剩余空间, 因此子视图应使用固定 dp 高度。 */
    fun scrollableRoot(): Pair<ScrollView, LinearLayout> {
        val inner = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }
        val scroll = ScrollView(context).apply { addView(inner) }
        return scroll to inner
    }

    private fun dp(value: Int): Int =
        (value * context.resources.displayMetrics.density + 0.5f).toInt()

    fun row(): LinearLayout = LinearLayout(context).apply {
        orientation = LinearLayout.HORIZONTAL
    }

    fun title(text: String): TextView = TextView(context).apply {
        this.text = text
        textSize = 28f
        gravity = Gravity.CENTER_HORIZONTAL
    }

    fun button(text: String, onClick: () -> Unit): Button = Button(context).apply {
        this.text = text
        setOnClickListener { onClick() }
    }

    fun input(hint: String, value: String = ""): EditText = EditText(context).apply {
        this.hint = hint
        setText(value)
        setSingleLine(true)
    }

    fun weightParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

    fun withLog(content: LinearLayout): ScreenWithLog {
        val contentScroll = ScrollView(context).apply { addView(content) }
        return buildSplitScreen(contentScroll)
    }

    /** 用于已自带 ScrollView 根(scrollableRoot)的屏幕。 */
    fun withLog(rootScroll: ScrollView): ScreenWithLog {
        return buildSplitScreen(rootScroll)
    }

    private fun buildSplitScreen(contentScroll: ScrollView): ScreenWithLog {
        val logView = TextView(context).apply { textSize = 13f }
        val logScroll = ScrollView(context).apply {
            isFillViewport = false
            isVerticalScrollBarEnabled = true
            scrollBarStyle = View.SCROLLBARS_INSIDE_INSET
            addView(logView)
        }
        val olderLogsButton = button("↑ 旧日志") { logScroll.smoothScrollBy(0, -dp(140)) }
        val latestLogsButton = button("↓ 最新") { logScroll.fullScroll(View.FOCUS_DOWN) }
        val logHeader = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(TextView(context).apply {
                text = "运行日志（可滚动）"
                textSize = 14f
                setPadding(dp(16), dp(6), dp(16), dp(4))
            }, LinearLayout.LayoutParams(0, -2, 1f))
            addView(olderLogsButton)
            addView(latestLogsButton)
        }
        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            addView(contentScroll, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(logHeader)
            addView(logScroll, LinearLayout.LayoutParams(-1, dp(180)))
        }
        return ScreenWithLog(root, logView, logScroll, olderLogsButton, latestLogsButton)
    }
}
