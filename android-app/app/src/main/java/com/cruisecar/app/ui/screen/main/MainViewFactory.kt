package com.cruisecar.app.ui.screen.main

import android.content.Context
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

data class ScreenWithLog(
    val root: LinearLayout,
    val logView: TextView
)

class MainViewFactory(private val context: Context) {
    fun rootLayout(): LinearLayout = LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(32, 32, 32, 32)
    }

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
        val logView = TextView(context).apply { textSize = 13f }
        val scroll = ScrollView(context).apply { addView(logView) }
        content.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        return ScreenWithLog(root = content, logView = logView)
    }
}
