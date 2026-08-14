package com.cruisecar.app.ui.screen

import com.cruisecar.app.connection.control.ControlClient
import com.cruisecar.app.connection.control.ControlServer
import com.cruisecar.app.connection.control.DiscoveryResponder
import com.cruisecar.app.connection.control.DiscoveryScanner
import com.cruisecar.app.connection.esp32.BluetoothSppClient
import com.cruisecar.app.connection.esp32.Esp32BlePairing
import com.cruisecar.app.connection.webrtc.WebRtcCall
import com.cruisecar.app.data.remote.RemoteApi
import com.cruisecar.app.data.remote.RemoteReceiver
import com.cruisecar.app.feature.follow.SmartFollowController
import com.cruisecar.app.protocol.ControlFrame
import com.cruisecar.app.protocol.ControlMode
import com.cruisecar.app.protocol.ControlProtocol
import com.cruisecar.app.protocol.GamepadState
import com.cruisecar.app.protocol.StatusCommand
import com.cruisecar.app.protocol.toHexLine
import com.cruisecar.app.ui.screen.main.MainViewFactory
import com.cruisecar.app.ui.widget.CameraPreviewView
import com.cruisecar.app.ui.widget.VideoGamepadView
import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.cruisecar.app.data.local.ReceiverIdentityStore
import com.cruisecar.app.domain.model.ConnectionMode
import com.cruisecar.app.domain.model.ReceiverIdentity
import com.cruisecar.app.mvi.AppIntent
import com.cruisecar.app.mvi.MainViewModel
import org.webrtc.SurfaceViewRenderer
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

private enum class RootTab { CONNECT, MINE }

private const val ROLE_SENDER = "sender"
private const val ROLE_RECEIVER = "receiver"
private val HOME_BG_TOP = Color.rgb(15, 23, 42)
private val HOME_BG_MID = Color.rgb(11, 36, 71)
private val HOME_BG_BOTTOM = Color.rgb(3, 21, 37)
private val HOME_CARD_BG = Color.argb(34, 255, 255, 255)
private val HOME_CARD_STROKE = Color.argb(110, 125, 211, 252)
private val HOME_TEXT_MUTED = Color.rgb(186, 230, 253)
private val HOME_TEXT_SOFT = Color.rgb(224, 242, 254)
private val HOME_ACCENT = Color.rgb(224, 242, 254)
private val HOME_ACCENT_TEXT = Color.rgb(8, 47, 73)

class ConnectPageFragment : android.app.Fragment() {
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View =
        (activity as MainActivity).createConnectFragmentView()
}

class MinePageFragment : android.app.Fragment() {
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View =
        (activity as MainActivity).createMineFragmentView()
}

class MainActivity : Activity() {
    private val tag = "CruiseCar"
    private val appVersionLabel = "v0.1.3-camera-switch"
    private val controlPort = 42101
    private val videoPort = 42102
    private val remoteControlPort = 42110
    private val remoteWebRtcPort = 42112
    private val remoteManagerPort = 8088
    private val defaultRemoteHost = "116.62.32.90"
    private val remoteTurnUrl = ""
    private val remoteTurnUser = ""
    private val remoteTurnPassword = ""
    private val controlClient = ControlClient()
    private val bluetooth = BluetoothSppClient()
    private val senderExecutor = Executors.newSingleThreadExecutor()
    /* 舵机发送独立线程: 与手柄发送隔离, 手柄流量再大也不会拖慢舵机实时性 */
    private val senderServoExecutor = Executors.newSingleThreadExecutor()
    private var discoveryResponder: DiscoveryResponder? = null
    private var controlServer: ControlServer? = null
    /* 接收端转发独立线程: 把蓝牙写从 TCP 读取线程挪开, 避免 ESP32 蓝牙写阻塞
       时反向压垮发送端 TCP(导致舵机/手柄指令在发送端排队、延迟累积)。 */
    private val forwardExecutor = Executors.newSingleThreadExecutor()
    /* 接收端状态回报 + ESP32 心跳的定时任务(1Hz) */
    private val receiverReporter: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private var cameraPreview: CameraPreviewView? = null
    private var senderVideoRenderer: SurfaceViewRenderer? = null
    private var senderVideoArea: FrameLayout? = null
    private var senderLayout: LinearLayout? = null
    private var receiverVideoRenderer: SurfaceViewRenderer? = null
    private var receiverLayout: LinearLayout? = null
    private var receiverEspStatusView: TextView? = null
    /* 接收端 → 发送端 状态回报定时任务句柄, stopReceiverServices 时取消 */
    private var receiverReporterTask: ScheduledFuture<*>? = null
    private var receiverServicesStarted = false
    private var smartFollow: SmartFollowController? = null
    private var webRtcCall: WebRtcCall? = null
    private lateinit var logView: TextView
    private var connectedReceiverHost: String? = null
    private lateinit var viewModel: MainViewModel
    private var senderMode = ControlMode.MANUAL
    private var receiverMode = ControlMode.MANUAL
    private var senderVideoEnabled = false
    private var senderVideoButton: Button? = null
    private var senderReceiverStatusView: TextView? = null
    private var senderCameraFacing = WebRtcCall.CameraFacing.FRONT
    private var senderCameraButton: Button? = null
    private var lastSenderState: GamepadState? = null
    private var lastSenderAtMs: Long = 0
    private var lastReceiverGamepadLogAtMs: Long = 0
    private var lastReceiverForwardLogAtMs: Long = 0
    /* 舵机命令合并: 只保留"最新目标角度", 同一时刻最多排 1 个发送任务,
       避免拖动竖向舵机控件时大量回调把队列撑爆导致延迟累积。发送走独立线程,
       不与手柄发送争用同一执行器, 降低端到端延迟。 */
    private var senderServoTargetAngle = 130
    private var senderServoSendScheduled = false
    private val senderServoLock = Any()
    private var backAction: (() -> Unit)? = null
    private lateinit var viewFactory: MainViewFactory
    private var selectedRootTab = RootTab.CONNECT
    private var homeContainerId: Int = View.generateViewId()
    private var loginDialogAutoShown = false
    private var loginDialog: AlertDialog? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewFactory = MainViewFactory(this)
        viewModel = MainViewModel(ReceiverIdentityStore(this))
        log("CruiseCar APK debug $appVersionLabel started")
        requestBasePermissions()
        showHomeScreen()
    }

    private fun showHomeScreen(tab: RootTab = selectedRootTab) {
        backAction = null
        val targetTab = if (tab == RootTab.CONNECT && !hasRemoteLogin()) RootTab.MINE else tab
        selectedRootTab = targetTab
        val layout = homeShellLayout()
        layout.addView(topNavigation(targetTab))
        val fragmentContainer = FrameLayout(this).apply { id = homeContainerId }
        layout.addView(fragmentContainer, LinearLayout.LayoutParams(-1, 0, 1f))
        layout.addView(bottomNavigation(targetTab))
        logView = TextView(this)
        setContentView(layout)
        showRootFragment(targetTab)
        if (!hasRemoteLogin() && !loginDialogAutoShown) {
            loginDialogAutoShown = true
            Handler(Looper.getMainLooper()).post { showLoginDialog() }
        }
    }

    private fun topNavigation(tab: RootTab): LinearLayout {
        val nav = row().apply {
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(18), dp(18), dp(8))
            background = homeGradient()
        }
        val brand = TextView(this).apply {
            text = "✦ CruiseCar"
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(HOME_TEXT_SOFT)
            gravity = Gravity.CENTER
            setPadding(dp(18), 0, dp(18), 0)
            background = glassBg(dp(24))
        }
        nav.addView(brand, LinearLayout.LayoutParams(-2, dp(48)))
        nav.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))
        val iconGroup = row().apply {
            gravity = Gravity.CENTER
            setPadding(dp(8), 0, dp(8), 0)
            background = roundedBg(Color.TRANSPARENT, dp(4))
        }
        if (tab == RootTab.CONNECT) {
            iconGroup.addView(topTextButton(if (viewModel.state.remotePreferredRole == ROLE_RECEIVER) "接收端" else "发送端") { togglePreferredRole() })
        } else {
            iconGroup.addView(topIconButton(if (hasRemoteLogin()) "↻" else "+") { showLoginDialog() })
            iconGroup.addView(topIconButton("⚙") { showSettingsScreen() })
        }
        nav.addView(iconGroup, LinearLayout.LayoutParams(-2, dp(56)))
        return nav
    }

    private fun bottomNavigation(tab: RootTab): LinearLayout {
        val holder = row().apply {
            gravity = Gravity.CENTER
            setPadding(dp(18), dp(8), dp(18), dp(18))
            background = homeGradient()
        }
        val pill = row().apply {
            gravity = Gravity.CENTER
            setPadding(dp(4), dp(4), dp(4), dp(4))
            background = roundedBg(Color.argb(36, 255, 255, 255), dp(34), strokeColor = Color.argb(58, 224, 242, 254))
        }
        pill.addView(bottomTabButton("连接", tab == RootTab.CONNECT) {
            showHomeScreen(RootTab.CONNECT)
        }, LinearLayout.LayoutParams(0, dp(56), 1f))
        pill.addView(centerDivider(), LinearLayout.LayoutParams(dp(1), dp(24)))
        pill.addView(bottomTabButton("我的", tab == RootTab.MINE) {
            showHomeScreen(RootTab.MINE)
        }, LinearLayout.LayoutParams(0, dp(56), 1f))
        holder.addView(pill, LinearLayout.LayoutParams(-1, dp(66)))
        return holder
    }

    private fun showRootFragment(tab: RootTab) {
        val fragment = if (tab == RootTab.CONNECT) ConnectPageFragment() else MinePageFragment()
        fragmentManager.beginTransaction()
            .replace(homeContainerId, fragment)
            .commitAllowingStateLoss()
    }

    internal fun createConnectFragmentView(): View {
        val layout = homePageLayout()
        addConnectTab(layout)
        return layout
    }

    internal fun createMineFragmentView(): View {
        val layout = homePageLayout()
        addMineTab(layout)
        return layout
    }

    private fun addConnectTab(layout: LinearLayout) {
        val state = viewModel.state
        layout.addView(pageTitle("连接", "在线设备控制 · 服务器账号模式"))
        layout.addView(connectHeroCard(state))
        if (!hasRemoteLogin()) {
            layout.addView(homeActionButton("去登录账号") { showHomeScreen(RootTab.MINE) })
            layout.addView(infoText("首次使用服务器模式需要先登录；登录后会保存账号 token，下次打开可直接进入控制。"))
            return
        }

        if (state.remotePreferredRole == ROLE_RECEIVER) {
            val identity = viewModel.receiverIdentity()
            viewModel.dispatch(AppIntent.SetRemoteDeviceId(identity.deviceId))
            layout.addView(deviceIdentityView(identity))
            layout.addView(homeActionButton("启动接收端") {
                enterServerReceiver(identity)
            })
        } else {
            layout.addView(lastDeviceInfoView())
            layout.addView(homeActionButton(if (state.remoteDeviceId.isBlank()) "选择接收端设备" else "连接设备") {
                if (state.remoteDeviceId.isBlank()) {
                    loadDevicesAndShowPicker()
                } else {
                    setConnectionMode(ConnectionMode.SERVER)
                    showSenderScreen()
                }
            })
            layout.addView(homeActionButton("切换接收端设备", primary = false) { loadDevicesAndShowPicker() })
        }
    }

    private fun addMineTab(layout: LinearLayout) {
        val state = viewModel.state
        layout.addView(pageTitle("我的", "账号、设备与偏好设置"))
        layout.addView(accountHeroCard(state))
        layout.addView(lastDeviceInfoView())
        if (hasRemoteLogin()) {
            layout.addView(homeActionButton("刷新设备信息", primary = false) { loadDevicesAndShowPicker(stayInMine = true) })
        } else {
            layout.addView(infoText("请点击右上角“登录”完成账号登录，登录后会保存 token，下次打开无需重新输入。"))
        }
    }

    private fun showLoginDialog() {
        if (loginDialog?.isShowing == true) return
        val state = viewModel.state
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 8, 32, 0)
        }
        val userInput = input("账号", state.remoteUsername)
        val passInput = input("密码", "").apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        form.addView(serverHostView())
        form.addView(userInput)
        form.addView(passInput)
        loginDialog = AlertDialog.Builder(this)
            .setTitle(if (hasRemoteLogin()) "重新登录" else "账号登录")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("登录", null)
            .create()
        loginDialog?.setOnShowListener { dialog ->
            val alert = dialog as AlertDialog
            alert.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val username = userInput.text.toString().trim()
                val password = passInput.text.toString()
                if (username.isBlank() || password.isBlank()) {
                    toast("请填写账号和密码")
                    return@setOnClickListener
                }
                alert.dismiss()
                loginFromInputs(username, password, navigateToConnect = true)
            }
        }
        loginDialog?.show()
    }

    private fun showSettingsScreen() {
        backAction = { showHomeScreen(RootTab.MINE) }
        val layout = rootLayout()
        layout.addView(title("设置"))
        layout.addView(button("调试台 (识别 + 遥控)") {
            startActivity(Intent(this, DebugActivity::class.java))
        })
        layout.addView(button("发送端 - 局域网1") { setConnectionMode(ConnectionMode.LAN); showSenderScreen() })
        layout.addView(button("接收端 - 局域网2") { setConnectionMode(ConnectionMode.LAN); showReceiverScreen() })
        layout.addView(button("返回我的") { showHomeScreen(RootTab.MINE) })
        setContentView(withLog(layout))
    }

    private fun loginFromInputs(username: String, password: String, navigateToConnect: Boolean) {
        Thread {
            try {
                val host = defaultRemoteHost
                if (host.isBlank() || username.isBlank() || password.isBlank()) {
                    toast("请填写服务器、账号和密码")
                    return@Thread
                }
                val managerBaseUrl = "http://$host:$remoteManagerPort"
                val token = RemoteApi.login(managerBaseUrl, username, password)
                viewModel.saveRemoteAccount(host, username, token, managerBaseUrl)
                setConnectionMode(ConnectionMode.SERVER)
                log("服务器账号登录成功: $username")
                runOnUiThread { showHomeScreen(if (navigateToConnect) RootTab.CONNECT else RootTab.MINE) }
            } catch (e: Exception) {
                log("服务器登录/配置失败: ${e.message}")
            }
        }.start()
    }

    private fun loadDevicesAndShowPicker(stayInMine: Boolean = false) {
        val state = viewModel.state
        if (!hasRemoteLogin()) {
            toast("请先登录账号")
            showHomeScreen(RootTab.MINE)
            return
        }
        Thread {
            try {
                val devices = RemoteApi.listReceivers(state.remoteManagerBaseUrl.ifBlank { "http://$defaultRemoteHost:$remoteManagerPort" }, state.remoteToken)
                if (devices.isEmpty()) {
                    log("账号下暂无接收端，请先用接收端加入")
                    if (stayInMine) runOnUiThread { showHomeScreen(RootTab.MINE) }
                } else {
                    runOnUiThread { showDevicePicker(devices, stayInMine) }
                }
            } catch (e: Exception) {
                log("获取设备列表失败: ${e.message}")
            }
        }.start()
    }

    private fun showDevicePicker(devices: List<RemoteReceiver>, stayInMine: Boolean = false) {
        backAction = { showHomeScreen(if (stayInMine) RootTab.MINE else RootTab.CONNECT) }
        val layout = rootLayout()
        layout.addView(title("选择接收端"))
        devices.forEach { device ->
            val row = row()
            row.addView(button("${device.name.ifBlank { device.deviceId }} | ${if (device.online) "在线" else "离线"} | ESP32=${device.espConnected}") {
                viewModel.dispatch(AppIntent.SetLastRemoteDevice(device.deviceId, device.name, device.online, device.espConnected, device.mode))
                log("已选择接收端: ${device.deviceId}")
                if (stayInMine) {
                    showHomeScreen(RootTab.MINE)
                } else {
                    setConnectionMode(ConnectionMode.SERVER)
                    showSenderScreen()
                }
            }, LinearLayout.LayoutParams(0, -2, 1f))
            row.addView(button("删除") { confirmDeleteReceiver(device, stayInMine) }, LinearLayout.LayoutParams(dp(86), -2).apply { leftMargin = dp(8) })
            layout.addView(row)
        }
        layout.addView(button("返回") { showHomeScreen(if (stayInMine) RootTab.MINE else RootTab.CONNECT) })
        setContentView(withLog(layout))
    }

    private fun confirmDeleteReceiver(device: RemoteReceiver, stayInMine: Boolean) {
        AlertDialog.Builder(this)
            .setTitle("删除接收端")
            .setMessage("确定从账号中删除 ${device.name.ifBlank { device.deviceId }}？离线设备可直接删除，在线设备需要先退出接收端。")
            .setNegativeButton("取消", null)
            .setPositiveButton("删除") { _, _ -> deleteReceiver(device, stayInMine) }
            .show()
    }

    private fun deleteReceiver(device: RemoteReceiver, stayInMine: Boolean) {
        val state = viewModel.state
        Thread {
            try {
                RemoteApi.deleteReceiver(state.remoteManagerBaseUrl.ifBlank { "http://$defaultRemoteHost:$remoteManagerPort" }, state.remoteToken, device.deviceId)
                if (state.remoteDeviceId == device.deviceId || state.lastRemoteDeviceId == device.deviceId) {
                    viewModel.dispatch(AppIntent.ClearLastRemoteDevice)
                }
                log("已删除接收端: ${device.deviceId}")
                loadDevicesAndShowPicker(stayInMine)
            } catch (e: Exception) {
                log("删除接收端失败: ${e.message}")
            }
        }.start()
    }

    private fun enterServerReceiver(identity: ReceiverIdentity) {
        val state = viewModel.state
        if (!hasRemoteLogin()) {
            showHomeScreen(RootTab.MINE)
            return
        }
        Thread {
            try {
                val managerBaseUrl = state.remoteManagerBaseUrl.ifBlank { "http://$defaultRemoteHost:$remoteManagerPort" }
                RemoteApi.addReceiver(managerBaseUrl, state.remoteToken, identity.deviceId, identity.displayName)
                viewModel.dispatch(AppIntent.SetLastRemoteDevice(identity.deviceId, identity.displayName, online = true, espConnected = bluetooth.isConnected(), mode = receiverMode.name.lowercase()))
                setConnectionMode(ConnectionMode.SERVER)
                log("接收端已加入账号: ${identity.displayName} (${identity.deviceId})")
                runOnUiThread { showReceiverScreen() }
            } catch (e: Exception) {
                log("接收端加入账号失败: ${e.message}")
            }
        }.start()
    }

    private fun infoText(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 14f
        setTextColor(HOME_TEXT_MUTED)
        setPadding(dp(18), dp(16), dp(18), dp(16))
        background = glassBg(dp(22))
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(12) }
    }

    private fun pageTitle(title: String, subtitle: String): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(6), dp(10), dp(6), dp(18))
        addView(TextView(this@MainActivity).apply {
            text = title
            textSize = 32f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
        })
        addView(TextView(this@MainActivity).apply {
            text = subtitle
            textSize = 15f
            setTextColor(HOME_TEXT_MUTED)
            setPadding(0, dp(4), 0, 0)
        })
    }

    private fun connectHeroCard(state: com.cruisecar.app.mvi.AppState): LinearLayout = glassCard().apply {
        addView(TextView(this@MainActivity).apply {
            text = if (hasRemoteLogin()) "${preferredRoleLabel()} · 准备就绪" else "等待登录"
            textSize = 26f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
        })
        addView(TextView(this@MainActivity).apply {
            text = "服务器 http://$defaultRemoteHost/\n${if (hasRemoteLogin()) "已登录 ${state.remoteUsername}" else "登录后自动保存 token，下次无需重复输入"}"
            textSize = 14f
            setTextColor(HOME_TEXT_MUTED)
            setPadding(0, dp(10), 0, dp(14))
        })
        val chips = row().apply { gravity = Gravity.CENTER_VERTICAL }
        chips.addView(statusChip(if (hasRemoteLogin()) "ONLINE" else "LOGIN", hasRemoteLogin()))
        chips.addView(statusChip(preferredRoleLabel(), true), LinearLayout.LayoutParams(-2, dp(30)).apply { leftMargin = dp(8) })
        addView(chips)
    }

    private fun accountHeroCard(state: com.cruisecar.app.mvi.AppState): LinearLayout = glassCard().apply {
        addView(TextView(this@MainActivity).apply {
            text = state.remoteUsername.ifBlank { "未登录" }
            textSize = 28f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
        })
        addView(TextView(this@MainActivity).apply {
            text = "Token：${if (state.remoteToken.isNotBlank()) "已保存" else "未保存"}\n发送端 ID：${state.remoteSenderId.ifBlank { "未生成" }}"
            textSize = 14f
            setTextColor(HOME_TEXT_MUTED)
            setPadding(0, dp(10), 0, 0)
        })
    }

    private fun glassCard(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(22), dp(20), dp(22), dp(20))
        background = glassBg(dp(28))
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(14) }
    }

    private fun statusChip(text: String, active: Boolean): TextView = TextView(this).apply {
        this.text = text
        textSize = 12f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(if (active) Color.rgb(5, 46, 22) else HOME_TEXT_SOFT)
        setPadding(dp(14), 0, dp(14), 0)
        background = roundedBg(if (active) Color.rgb(34, 197, 94) else Color.argb(34, 255, 255, 255), dp(15), strokeColor = Color.argb(70, 224, 242, 254))
        layoutParams = LinearLayout.LayoutParams(-2, dp(30))
    }

    private fun homeActionButton(text: String, primary: Boolean = true, onClick: () -> Unit): TextView = TextView(this).apply {
        this.text = text
        textSize = 17f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(if (primary) HOME_ACCENT_TEXT else HOME_TEXT_SOFT)
        setPadding(dp(16), 0, dp(16), 0)
        background = roundedBg(if (primary) HOME_ACCENT else Color.argb(34, 255, 255, 255), dp(28), strokeColor = if (primary) HOME_ACCENT else HOME_CARD_STROKE)
        layoutParams = LinearLayout.LayoutParams(-1, dp(56)).apply { bottomMargin = dp(12) }
        setOnClickListener { onClick() }
    }

    private fun homeShellLayout(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        background = homeGradient()
    }

    private fun homePageLayout(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(6), dp(20), dp(10))
        background = homeGradient()
    }

    private fun topTextButton(text: String, onClick: () -> Unit): TextView = TextView(this).apply {
        this.text = text
        textSize = 14f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(HOME_TEXT_SOFT)
        setPadding(dp(16), 0, dp(16), 0)
        background = glassBg(dp(22))
        setOnClickListener { onClick() }
    }.also {
        it.layoutParams = LinearLayout.LayoutParams(-2, dp(44)).apply { leftMargin = dp(8) }
    }

    private fun topIconButton(text: String, onClick: () -> Unit): TextView = TextView(this).apply {
        this.text = text
        textSize = 24f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(HOME_TEXT_SOFT)
        background = glassBg(dp(22))
        setOnClickListener { onClick() }
    }.also {
        it.layoutParams = LinearLayout.LayoutParams(dp(48), dp(48)).apply { leftMargin = dp(8) }
    }

    private fun centerDivider(): View = View(this).apply {
        setBackgroundColor(Color.argb(95, 224, 242, 254))
    }

    private fun bottomTabButton(label: String, selected: Boolean, onClick: () -> Unit): TextView = TextView(this).apply {
        text = label
        textSize = 17f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        setTextColor(if (selected) HOME_ACCENT_TEXT else HOME_TEXT_MUTED)
        background = if (selected) roundedBg(HOME_ACCENT, dp(28)) else roundedBg(Color.TRANSPARENT, dp(28))
        setOnClickListener { onClick() }
    }

    private fun homeGradient(): GradientDrawable = GradientDrawable(
        GradientDrawable.Orientation.TL_BR,
        intArrayOf(HOME_BG_TOP, HOME_BG_MID, HOME_BG_BOTTOM)
    )

    private fun glassBg(radius: Int): GradientDrawable = roundedBg(HOME_CARD_BG, radius, HOME_CARD_STROKE)

    private fun roundedBg(color: Int, radius: Int, strokeColor: Int? = null): GradientDrawable =
        GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            setColor(color)
            cornerRadius = radius.toFloat()
            strokeColor?.let { setStroke(dp(1), it) }
        }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()

    private fun lastDeviceInfoView(): TextView {
        val state = viewModel.state
        val deviceText = if (state.lastRemoteDeviceId.isBlank()) {
            "上次连接设备：无"
        } else {
            "上次连接设备：${state.lastRemoteDeviceName.ifBlank { state.lastRemoteDeviceId }}\n设备 ID：${state.lastRemoteDeviceId}\n状态：${if (state.lastRemoteDeviceOnline) "在线" else "离线"} | ESP32=${state.lastRemoteDeviceEspConnected} | 模式=${state.lastRemoteDeviceMode}"
        }
        return infoText(deviceText)
    }

    private fun hasRemoteLogin(): Boolean =
        viewModel.state.remoteUsername.isNotBlank() && viewModel.state.remoteToken.isNotBlank()

    private fun preferredRoleLabel(): String =
        if (viewModel.state.remotePreferredRole == ROLE_RECEIVER) "接收端" else "发送端"

    private fun roleSwitchText(): String = "切换为${if (viewModel.state.remotePreferredRole == ROLE_RECEIVER) "发送端" else "接收端"}"

    private fun togglePreferredRole() {
        val nextRole = if (viewModel.state.remotePreferredRole == ROLE_RECEIVER) ROLE_SENDER else ROLE_RECEIVER
        viewModel.dispatch(AppIntent.SetRemotePreferredRole(nextRole))
        showHomeScreen(RootTab.CONNECT)
    }

    private fun deviceIdentityView(identity: ReceiverIdentity): TextView = TextView(this).apply {
        text = "接收端名称：${identity.displayName}\n接收端设备ID：${identity.deviceId}\n\n接收端会自动使用该 ID 加入服务器；发送端登录同一账号后可直接选择设备。"
        textSize = 14f
        setTextColor(HOME_TEXT_MUTED)
        setPadding(dp(18), dp(16), dp(18), dp(16))
        background = glassBg(dp(22))
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(12) }
    }

    private fun serverHostView(): TextView = TextView(this).apply {
        text = "服务器：http://$defaultRemoteHost/"
        textSize = 14f
        setTextColor(Color.rgb(70, 70, 70))
        setPadding(0, 12, 0, 12)
    }

    private fun setConnectionMode(mode: ConnectionMode) {
        viewModel.dispatch(AppIntent.SetConnectionMode(mode))
    }

    private fun showSenderScreen() {
        backAction = {
            releaseSenderCall()
            senderVideoArea = null
            senderLayout = null
            showHomeScreen(RootTab.CONNECT)
        }
        val layout = rootLayout()
        senderLayout = layout
        layout.addView(title("发送端多模式控制"))

        val videoArea = VideoGamepadView(this)
        senderVideoArea = videoArea
        val remoteVideo = createSenderVideoRenderer(videoArea)
        videoArea.onStateChanged = stateChanged@{ state ->
            if (senderMode != ControlMode.MANUAL) {
                /* 用户在非手动模式下操作了摇杆(lx/ly 有实际输入), 自动退出智能模式回到手动遥控 */
                val controlling = state.lx != 128 || state.ly != 128 || state.buttons != 0
                if (!controlling) return@stateChanged
                val exited = senderMode
                senderMode = ControlMode.MANUAL
                sendMode(ControlMode.MANUAL)
                toast("已退出 ${exited.label} 模式，切换为手动遥控")
            }
            if (senderMode == ControlMode.MANUAL) {
                sendFromGamepad(state)
            }
        }
        videoArea.onServoChanged = { angle -> sendServoToReceiver(angle) }
        layout.addView(videoArea, LinearLayout.LayoutParams(-1, 0, 2.4f))

        /* 接收端状态(ESP32 连接态 + 接收端当前模式)由接收端 1Hz 回传, 此处展示 */
        senderReceiverStatusView = TextView(this).apply {
            text = "接收端：未连接"
            textSize = 14f
            gravity = Gravity.CENTER_HORIZONTAL
            setTextColor(Color.rgb(200, 60, 60))
        }
        layout.addView(senderReceiverStatusView)

        /* 发送端被动感知接收端状态; 接收端断开时回退 UI */
        controlClient.onStatus = { espConnected, mode ->
            runOnUiThread {
                senderReceiverStatusView?.apply {
                    text = "ESP32：${if (espConnected) "已连接" else "未连接"}  |  接收端模式：${mode.label}"
                    setTextColor(if (espConnected) Color.rgb(60, 180, 60) else Color.rgb(200, 60, 60))
                }
            }
        }
        controlClient.onReceiverGone = {
            runOnUiThread {
                senderReceiverStatusView?.apply {
                    text = "接收端：已断开"
                    setTextColor(Color.rgb(200, 60, 60))
                }
            }
        }

        layout.addView(button(senderConnectButtonText()) { connectSenderByMode() })

        layout.addView(button("远程连接 ESP32") {
            if (controlClient.isConnected()) {
                controlClient.sendCommand(ControlProtocol.CMD_CONNECT_ESP32)
                log("已向接收端下发指令：自动扫描并连接 ESP32")
            } else {
                log("未连接接收端，无法远程触发 ESP32 连接")
            }
        })

        val modeRow = row()
        modeRow.addView(button("实时视频") { startSenderVideoMode() }, weightParams())
        modeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        modeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(modeRow)
        modeRow.visibility = View.GONE
        val driveModeRow = row()
        driveModeRow.addView(button("智能跟随") { sendMode(ControlMode.SMART_FOLLOW, remoteVideo) }, weightParams())
        driveModeRow.addView(button("智能巡逻") { sendMode(ControlMode.PATROL, remoteVideo) }, weightParams())
        layout.addView(driveModeRow)
        senderVideoButton = button(senderVideoButtonText()) { toggleSenderVideo() }
        layout.addView(senderVideoButton)
        senderCameraButton = button(senderCameraButtonText()) { toggleSenderCamera() }
        layout.addView(senderCameraButton)

        layout.addView(button("停止 / 回中") { sendFromGamepad(GamepadState(), force = true) })
        layout.addView(button("返回") {
            backAction?.invoke()
        })
        setContentView(withLog(layout))
    }

    private fun showReceiverScreen() {
        backAction = { stopReceiverServices(); receiverLayout = null; showHomeScreen(RootTab.CONNECT) }
        val layout = rootLayout()
        receiverLayout = layout
        layout.addView(title("接收端多模式中转"))

        val preview = CameraPreviewView(this)
        cameraPreview = preview
        layout.addView(preview, LinearLayout.LayoutParams(-1, 0, 1.5f))

        receiverEspStatusView = TextView(this).apply {
            text = "未连接"
            textSize = 14f
            gravity = Gravity.CENTER_HORIZONTAL
            setTextColor(Color.rgb(200, 60, 60))
        }
        layout.addView(receiverEspStatusView)

        val blePairing = Esp32BlePairing(
            this, bluetooth,
            onLog = { log(it) },
            onToast = { msg -> Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() },
            onConnected = { updateReceiverEspStatus(true) }
        )

        layout.addView(button("自动扫描并连接 ESP32") { connectEsp32ByScan() })
        layout.addView(button("BLE 扫描附近设备") { blePairing.start() })

        setContentView(withLog(layout))
        preview.start()
        startReceiverServices()
    }

    private fun updateReceiverEspStatus(connected: Boolean) {
        receiverEspStatusView?.apply {
            text = if (connected) "已连接" else "未连接"
            setTextColor(if (connected) Color.rgb(60, 180, 60) else Color.rgb(200, 60, 60))
        }
    }

    private fun connectEsp32ByScan() {
        Thread {
            try {
                bluetooth.connectFirstByName(this, BluetoothSppClient.ESP32_DEVICE_NAME) { log(it) }
                log("ESP32 auto-connect ready")
                runOnUiThread { updateReceiverEspStatus(true) }
            } catch (e: Exception) {
                log("ESP32 auto-connect failed: ${e.message}")
                runOnUiThread { updateReceiverEspStatus(false) }
            }
        }.start()
    }

    private fun startReceiverServices() {
        if (receiverServicesStarted) return
        receiverServicesStarted = true
        val state = viewModel.state
        if (state.connectionMode != ConnectionMode.LAN) {
            startRemoteReceiverServices()
            return
        }
        discoveryResponder = DiscoveryResponder(controlPort).also { it.start { msg -> log(msg) } }
        controlServer = ControlServer(controlPort) { packet, frame ->
            handleReceiverControlFrame(packet, frame)
        }.also { it.start { msg -> log(msg) } }

        /* 1Hz 向所有已连接发送端回传状态(ESP32 连接态 + 接收端当前模式),
           这样发送端能被动感知接收端及 ESP32 的状态, 无需轮询。 */
        receiverReporterTask = receiverReporter.scheduleAtFixedRate({
            try {
                controlServer?.send(StatusCommand.packet(bluetooth.isConnected(), receiverMode))
            } catch (e: Exception) {
                log("Status report failed: ${e.message}")
            }
        }, 0, 1, TimeUnit.SECONDS)

        log("Receiver services ready: control=$controlPort webrtc-signal=$videoPort")
    }

    private fun startRemoteReceiverServices() {
        val state = viewModel.state
        controlClient.onFrame = { packet, frame -> handleReceiverControlFrame(packet, frame) }
        controlClient.onReceiverGone = {
            log("Remote receiver disconnected; waiting for retry")
        }
        receiverReporterTask = receiverReporter.scheduleAtFixedRate({
            try {
                if (!controlClient.isConnected()) {
                    try {
                        log("Remote receiver reconnecting: ${state.remoteHost}:$remoteControlPort device=${state.remoteDeviceId}")
                        controlClient.connectRemoteReceiver(state.remoteHost, remoteControlPort, state.remoteDeviceId, state.remoteToken) { log(it) }
                        log("Remote receiver ready: device=${state.remoteDeviceId} server=${state.remoteHost}")
                    } catch (e: Exception) {
                        log("Remote receiver retry failed: ${e.message}")
                    }
                }
                if (controlClient.isConnected()) {
                    controlClient.sendRaw(StatusCommand.packet(bluetooth.isConnected(), receiverMode))
                }
            } catch (e: Exception) {
                log("Remote status report failed: ${e.message}")
            }
        }, 0, 3, TimeUnit.SECONDS)
    }

    private fun handleReceiverControlFrame(packet: ByteArray, frame: ControlFrame) {
        when (frame) {
            is ControlFrame.Gamepad -> {
                if (receiverMode == ControlMode.MANUAL) {
                    logReceiverGamepad(frame.state)
                    forwardToEsp32(packet, "Forwarded: ${packet.toHexLine()}")
                } else {
                    logReceiverGamepad(frame.state, "ignored in ${receiverMode.label}")
                }
            }
            is ControlFrame.Servo -> forwardToEsp32(packet, "Forwarded servo: idx=${frame.index} angle=${frame.angle}")
            is ControlFrame.Mode -> runOnUiThread {
                when (frame.mode) {
                    ControlMode.VIDEO_CALL -> setReceiverVideoEnabled(true)
                    ControlMode.VIDEO_OFF -> setReceiverVideoEnabled(false)
                    else -> applyReceiverDriveMode(frame.mode)
                }
            }
            is ControlFrame.Command -> {
                if (frame.code == ControlProtocol.CMD_CONNECT_ESP32) {
                    log("收到发送端指令：远程连接 ESP32")
                    connectEsp32ByScan()
                }
            }
            is ControlFrame.Status -> Unit
        }
    }

    private fun forwardToEsp32(packet: ByteArray, message: String) {
        if (bluetooth.isConnected()) {
            forwardExecutor.execute {
                if (bluetooth.send(packet)) {
                    log(message)
                } else {
                    logReceiverForwardStatus("ESP32 send failed: ${packet.toHexLine()}")
                    runOnUiThread { updateReceiverEspStatus(false) }
                }
            }
        } else {
            logReceiverForwardStatus("ESP32 not connected; received control frame but not forwarded: ${packet.toHexLine()}")
            runOnUiThread { updateReceiverEspStatus(false) }
        }
    }

    private fun logReceiverGamepad(state: GamepadState, suffix: String = "received") {
        val now = System.currentTimeMillis()
        if (now - lastReceiverGamepadLogAtMs < 800) return
        lastReceiverGamepadLogAtMs = now
        log("Receiver gamepad $suffix: lx=${state.lx} ly=${state.ly} rx=${state.rx} ry=${state.ry} buttons=${state.buttons}")
    }

    private fun logReceiverForwardStatus(message: String) {
        val now = System.currentTimeMillis()
        if (now - lastReceiverForwardLogAtMs < 1000) return
        lastReceiverForwardLogAtMs = now
        log(message)
    }

    private fun stopReceiverServices() {
        smartFollow?.stop()
        smartFollow = null
        receiverReporterTask?.cancel(false)
        receiverReporterTask = null
        discoveryResponder?.stop()
        controlServer?.stop()
        controlClient.onFrame = null
        controlClient.onReceiverGone = null
        controlClient.close()
        discoveryResponder = null
        controlServer = null
        cameraPreview?.stop()
        cameraPreview?.visibility = View.VISIBLE
        releaseReceiverCall()
        receiverLayout = null
        receiverServicesStarted = false
    }

    private fun applyReceiverDriveMode(mode: ControlMode) {
        receiverMode = mode
        smartFollow?.stop()
        smartFollow = null
        log("Receiver drive mode: ${mode.label}")

        when (mode) {
            ControlMode.MANUAL -> {
                bluetooth.send(GamepadState().toPacket())
            }
            ControlMode.SMART_FOLLOW -> {
                if (webRtcCall == null) {
                    cameraPreview?.visibility = View.VISIBLE
                    cameraPreview?.start()
                } else {
                    log("Smart follow selected while video is active; keep WebRTC video view")
                }
                smartFollow = SmartFollowController(
                    frameProvider = { cameraPreview?.snapshot(240, 180) },
                    onState = { state ->
                        try {
                            bluetooth.send(state.toPacket())
                        } catch (e: Exception) {
                            log("Smart follow send failed: ${e.message}")
                        }
                    }
                ).also { it.start { msg -> log(msg) } }
            }
            ControlMode.PATROL -> {
                bluetooth.send(GamepadState().toPacket())
                log("Patrol mode selected; route planner is reserved for the next phase")
            }
            ControlMode.VIDEO_CALL, ControlMode.VIDEO_OFF -> Unit
        }
    }

    private fun setReceiverVideoEnabled(enabled: Boolean) {
        log("Receiver video ${if (enabled) "enabled" else "disabled"}")
        if (!enabled) {
            releaseReceiverCall()
            if (receiverMode != ControlMode.SMART_FOLLOW) {
                cameraPreview?.visibility = View.VISIBLE
                cameraPreview?.start()
            }
            return
        }
        if (webRtcCall != null) {
            log("Receiver video already active; restart for latest sender")
            releaseReceiverCall()
        }
        if (receiverMode != ControlMode.SMART_FOLLOW) {
            cameraPreview?.stop()
            cameraPreview?.visibility = View.GONE
        }
        val renderer = createReceiverVideoRenderer()
        webRtcCall = WebRtcCall(
            this,
            WebRtcCall.Role.CALLER,
            renderer,
            cameraFacing = WebRtcCall.CameraFacing.FRONT,
            turnConfig = remoteTurnConfig(),
            onPeerDisconnected = {
                runOnUiThread {
                    log("Receiver video peer disconnected; closing current WebRTC call")
                    releaseReceiverCall()
                    if (receiverMode != ControlMode.SMART_FOLLOW) {
                        cameraPreview?.visibility = View.VISIBLE
                        cameraPreview?.start()
                    }
                }
            }
        ) { msg -> log(msg) }
        val state = viewModel.state
        if (state.connectionMode == ConnectionMode.LAN) {
            log("Receiver video signaling starting on $videoPort")
            webRtcCall?.startServer(videoPort)
        } else {
            log("Receiver video relay target ${state.remoteHost}:$remoteWebRtcPort room=${state.remoteDeviceId}")
            webRtcCall?.connectRelay(state.remoteHost, remoteWebRtcPort, state.remoteDeviceId, WebRtcCall.Role.CALLER, state.remoteToken)
        }
    }

    private fun senderConnectButtonText(): String = when (viewModel.state.connectionMode) {
        ConnectionMode.LAN -> "扫描接收端并连接"
        ConnectionMode.SERVER -> "连接已选设备"
    }

    private fun connectSenderByMode() {
        Thread {
            try {
                val state = viewModel.state
                if (state.connectionMode == ConnectionMode.LAN) {
                    val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                    val receivers = DiscoveryScanner(wifi).scan { log(it) }
                    if (receivers.isNotEmpty()) {
                        val receiver = receivers.first()
                        connectedReceiverHost = receiver.host
                        controlClient.connect(receiver.host, receiver.port) { log(it) }
                        sendFromGamepad(GamepadState(), force = true)
                    } else {
                        log("No receiver found")
                    }
                } else {
                    connectedReceiverHost = state.remoteHost
                    val senderId = if (state.remoteSenderId.isNotBlank()) state.remoteSenderId else "phone-${System.currentTimeMillis() % 100000}"
                    controlClient.connectRemoteSender(state.remoteHost, remoteControlPort, senderId, state.remoteDeviceId, state.remoteToken) { log(it) }
                    sendFromGamepad(GamepadState(), force = true)
                }
            } catch (e: Exception) {
                log("Connect failed: ${e.message}")
            }
        }.start()
    }

    private fun releaseReceiverCall() {
        webRtcCall?.release()
        webRtcCall = null
        receiverVideoRenderer?.let { renderer ->
            receiverLayout?.removeView(renderer)
        }
        receiverVideoRenderer = null
    }

    private fun createReceiverVideoRenderer(): SurfaceViewRenderer {
        val renderer = SurfaceViewRenderer(this).apply { visibility = View.VISIBLE }
        receiverVideoRenderer = renderer
        receiverLayout?.addView(renderer, LinearLayout.LayoutParams(-1, 0, 1.5f))
        return renderer
    }

    private fun sendMode(mode: ControlMode, remoteVideo: SurfaceViewRenderer? = null) {
        senderMode = mode
        senderExecutor.execute {
            try {
                log("sendMode ${mode.label} begin on ${threadName()}")
                if (!controlClient.isConnected()) {
                    log("Not connected: scan and connect receiver first")
                    return@execute
                }
                controlClient.sendMode(mode)
                log("Mode sent: ${mode.label}")
                if (mode != ControlMode.MANUAL) sendFromGamepad(GamepadState(), force = true)
            } catch (e: Throwable) {
                logError("Mode send failed", e)
            }
        }
    }

    private fun toggleSenderVideo() {
        if (senderVideoEnabled) {
            stopSenderVideoMode()
        } else {
            startSenderVideoMode()
        }
    }

    private fun startSenderVideoMode() {
        try {
            log("startSenderVideoMode begin on ${threadName()}")
            if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
                log("startSenderVideoMode repost to UI from ${threadName()}")
                runOnUiThread { startSenderVideoMode() }
                return
            }

            val state = viewModel.state
            val host = if (state.connectionMode == ConnectionMode.LAN) connectedReceiverHost else state.remoteHost
            if (!controlClient.isConnected() || host.isNullOrBlank()) {
                log("Not connected: scan and connect receiver first")
                return
            }

            Thread {
                try {
                    log("VIDEO_CALL mode packet send begin on ${threadName()}")
                    controlClient.sendMode(ControlMode.VIDEO_CALL)
                    log("Video command sent: ${ControlMode.VIDEO_CALL.label}")
                } catch (e: Throwable) {
                    logError("Video command send failed", e)
                }
            }.start()

            senderVideoEnabled = true
            senderVideoButton?.text = senderVideoButtonText()
            log("Sender video signaling target $host:${if (state.connectionMode == ConnectionMode.LAN) videoPort else remoteWebRtcPort}")
            log("releaseSenderCall before video on ${threadName()}")
            releaseSenderCall()
            log("createSenderVideoRenderer before video on ${threadName()}")
            val renderer = createSenderVideoRenderer()
            log("create WebRtcCall ANSWERER on ${threadName()}")
            webRtcCall = WebRtcCall(
                this,
                WebRtcCall.Role.ANSWERER,
                renderer,
                cameraFacing = senderCameraFacing,
                turnConfig = remoteTurnConfig()
            ) { msg -> log(msg) }
            log("WebRtcCall connect call on ${threadName()}")
            if (state.connectionMode == ConnectionMode.LAN) {
                webRtcCall?.connect(host, videoPort)
            } else {
                webRtcCall?.connectRelay(host, remoteWebRtcPort, state.remoteDeviceId, WebRtcCall.Role.ANSWERER, state.remoteToken)
            }
        } catch (e: Throwable) {
            logError("startSenderVideoMode failed", e)
        }
    }

    private fun releaseSenderCall() {
        log("releaseSenderCall on ${threadName()}")
        webRtcCall?.release()
        webRtcCall = null
        senderVideoRenderer?.let { renderer ->
            (renderer.parent as? ViewGroup)?.removeView(renderer)
        }
        senderVideoRenderer = null
    }

    private fun remoteTurnConfig(): WebRtcCall.TurnConfig =
        WebRtcCall.TurnConfig(
            url = remoteTurnUrl,
            username = remoteTurnUser,
            credential = remoteTurnPassword
        )

    private fun createSenderVideoRenderer(parent: FrameLayout? = null): SurfaceViewRenderer {
        log("createSenderVideoRenderer on ${threadName()}")
        val renderer = SurfaceViewRenderer(this)
        senderVideoRenderer = renderer
        val target = parent ?: senderVideoArea
        if (target != null) {
            target.addView(renderer, 0, FrameLayout.LayoutParams(-1, -1))
        } else {
            senderLayout?.addView(renderer, LinearLayout.LayoutParams(-1, 0, 1.1f))
        }
        return renderer
    }

    private fun toggleSenderCamera() {
        senderCameraFacing = if (senderCameraFacing == WebRtcCall.CameraFacing.FRONT) {
            WebRtcCall.CameraFacing.BACK
        } else {
            WebRtcCall.CameraFacing.FRONT
        }
        senderCameraButton?.text = senderCameraButtonText()
        log("Sender camera switched to ${senderCameraFacing.name.lowercase()}")
        if (senderVideoEnabled && webRtcCall != null) {
            startSenderVideoMode()
        }
    }

    private fun stopSenderVideoMode() {
        senderVideoEnabled = false
        senderVideoButton?.text = senderVideoButtonText()
        Thread {
            try {
                controlClient.sendMode(ControlMode.VIDEO_OFF)
                log("Video command sent: ${ControlMode.VIDEO_OFF.label}")
            } catch (e: Throwable) {
                logError("Video command send failed", e)
            }
        }.start()
        releaseSenderCall()
    }

    private fun senderVideoButtonText(): String =
        if (senderVideoEnabled) "视频：开" else "视频：关"

    private fun senderCameraButtonText(): String =
        if (senderCameraFacing == WebRtcCall.CameraFacing.FRONT) {
            "发送端摄像头：前置"
        } else {
            "发送端摄像头：后置"
        }

    private fun sendFromGamepad(state: GamepadState, force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && state == lastSenderState && now - lastSenderAtMs < 80) return
        if (!force && now - lastSenderAtMs < 24) return
        lastSenderState = state
        lastSenderAtMs = now

        senderExecutor.execute {
            try {
                if (controlClient.isConnected()) {
                    controlClient.send(state)
                    log("TCP sent: ${state.toHexLine()}")
                } else {
                    log("Not connected: scan and connect receiver first")
                }
            } catch (e: Exception) {
                log("Send failed: ${e.message}")
            }
        }
    }

    private fun sendServoToReceiver(angle: Int) {
        /* 合并: 只记录最新角度; 若已有待发送的 flush 任务则跳过排程,
           等该任务发送时自然会带上最新值。这样队列里最多 1 个待发任务,
           延迟被限制在"一次 TCP 写"以内, 不会随拖动时长累积。 */
        val shouldSchedule: Boolean
        synchronized(senderServoLock) {
            senderServoTargetAngle = angle
            shouldSchedule = !senderServoSendScheduled
            senderServoSendScheduled = true
        }
        if (!shouldSchedule) return

        senderServoExecutor.execute {
            val target = synchronized(senderServoLock) {
                senderServoSendScheduled = false
                senderServoTargetAngle
            }
            try {
                if (controlClient.isConnected()) {
                    controlClient.sendServo(angle = target)
                    log("舵机指令已发送: idx=0 angle=$target")
                } else {
                    log("未连接接收端: 请先扫描并连接接收端")
                }
            } catch (e: Exception) {
                log("舵机发送失败: ${e.message}")
            }
        }
    }

    private fun requestBasePermissions() {
        val permissions = mutableListOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.CHANGE_WIFI_MULTICAST_STATE,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        )
        if (android.os.Build.VERSION.SDK_INT >= 31) {
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
        }
        val missing = permissions.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), 100)
    }

    private fun rootLayout(): LinearLayout = viewFactory.rootLayout()

    private fun row(): LinearLayout = viewFactory.row()

    private fun title(text: String): TextView = viewFactory.title(text)

    private fun button(text: String, onClick: () -> Unit): Button = viewFactory.button(text, onClick)

    private fun input(hint: String, value: String = ""): EditText = viewFactory.input(hint, value)

    private fun weightParams(): LinearLayout.LayoutParams = viewFactory.weightParams()

    private fun withLog(content: LinearLayout): LinearLayout {
        val screen = viewFactory.withLog(content)
        logView = screen.logView
        return screen.root
    }

    private fun log(message: String) {
        Log.i(tag, message)
        runOnUiThread {
            if (::logView.isInitialized) {
                logView.append("$message\n")
            } else {
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }

    private fun logError(message: String, error: Throwable) {
        Log.e(tag, "$message on ${threadName()}: ${error.message}", error)
        log("$message: ${error.message}")
    }

    private fun threadName(): String =
        "${Thread.currentThread().name}/${Thread.currentThread().id}"

    override fun onBackPressed() {
        if (backAction != null) {
            backAction!!.invoke()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        stopReceiverServices()
        controlClient.close()
        bluetooth.close()
        webRtcCall?.release()
        senderExecutor.shutdownNow()
        senderServoExecutor.shutdownNow()
        forwardExecutor.shutdownNow()
        receiverReporter.shutdownNow()
        super.onDestroy()
    }
}
