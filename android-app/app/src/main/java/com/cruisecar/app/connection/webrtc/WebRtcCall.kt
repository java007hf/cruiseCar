package com.cruisecar.app.connection.webrtc

import android.content.Context
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import org.json.JSONObject
import org.webrtc.AudioSource
import org.webrtc.AudioTrack
import org.webrtc.Camera1Enumerator
import org.webrtc.Camera2Enumerator
import org.webrtc.CameraEnumerator
import org.webrtc.DataChannel
import org.webrtc.DefaultVideoDecoderFactory
import org.webrtc.DefaultVideoEncoderFactory
import org.webrtc.EglBase
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.MediaStream
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RtpReceiver
import org.webrtc.SdpObserver
import org.webrtc.SessionDescription
import org.webrtc.SurfaceTextureHelper
import org.webrtc.SurfaceViewRenderer
import org.webrtc.VideoCapturer
import org.webrtc.VideoSource
import org.webrtc.VideoTrack
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class WebRtcCall(
    private val context: Context,
    private val role: Role,
    private val renderer: SurfaceViewRenderer,
    private val cameraFacing: CameraFacing = CameraFacing.BACK,
    private val iceServerProvider: () -> List<IceServerConfig> = { emptyList() },
    private val onPeerDisconnected: () -> Unit = {},
    private val onLog: (String) -> Unit
) {
    enum class Role { CALLER, ANSWERER }
    enum class CameraFacing { FRONT, BACK }
    data class IceServerConfig(
        val urls: List<String>,
        val username: String = "",
        val credential: String = ""
    )

    private val eglBase = EglBase.create()
    private var factory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var signal: SignalChannel? = null
    private var serverSocket: ServerSocket? = null
    private var localCapturer: VideoCapturer? = null
    private var localSource: VideoSource? = null
    private var audioSource: AudioSource? = null
    private var audioManager: AudioManager? = null
    private var previousAudioMode: Int? = null
    private var previousSpeakerphoneOn: Boolean? = null
    private val pendingRemoteCandidates = mutableListOf<IceCandidate>()
    private val running = AtomicBoolean(false)
    private val connectionGeneration = AtomicInteger(0)

    init {
        onLog("WebRTC $role created on ${threadName()}")
    }

    fun startServer(port: Int) {
        onLog("WebRTC startServer on ${threadName()}")
        val generation = beginConnection()
        initRenderer()
        Thread {
            try {
                serverSocket = ServerSocket(port)
                onLog("WebRTC signaling server started on $port")
                val socket = serverSocket?.accept() ?: return@Thread
                if (!isCurrentConnection(generation)) {
                    socket.close()
                    return@Thread
                }
                onLog("WebRTC peer connected: ${socket.inetAddress.hostAddress}")
                Thread.sleep(1000)
                startPeer(SignalChannel(socket), createOffer = true, generation = generation)
            } catch (e: Exception) {
                if (running.get()) onLog("WebRTC server failed: ${e.message}")
            }
        }.start()
    }

    fun connect(host: String, port: Int) {
        onLog("WebRTC connect requested on ${threadName()}")
        val generation = beginConnection()
        initRenderer()
        Thread {
            var lastError: Exception? = null
            try {
                Thread.sleep(800)
                repeat(30) { attempt ->
                    if (!isCurrentConnection(generation)) return@Thread
                    try {
                        onLog("WebRTC signaling connect attempt ${attempt + 1}: $host:$port")
                        val socket = Socket(host, port)
                        if (!isCurrentConnection(generation)) {
                            socket.close()
                            return@Thread
                        }
                        onLog("WebRTC signaling connected: $host:$port")
                        startPeer(SignalChannel(socket), createOffer = false, generation = generation)
                        return@Thread
                    } catch (e: Exception) {
                        lastError = e
                        onLog("WebRTC signaling connect retry: ${e.message}")
                        Thread.sleep(300L)
                    }
                }
            } catch (e: Exception) {
                lastError = e
            }
            if (running.get()) {
                onLog("WebRTC connect failed: ${lastError?.message}")
            }
        }.start()
    }

    fun connectRelay(host: String, port: Int, roomId: String, signalRole: Role, token: String = "") {
        onLog("WebRTC relay connect requested room=$roomId role=$signalRole")
        val generation = beginConnection()
        initRenderer()
        Thread {
            var lastError: Exception? = null
            try {
                repeat(30) { attempt ->
                    if (!isCurrentConnection(generation)) return@Thread
                    try {
                        onLog("WebRTC relay connect attempt ${attempt + 1}: $host:$port")
                        val socket = Socket(host, port)
                        if (!isCurrentConnection(generation)) {
                            socket.close()
                            return@Thread
                        }
                        val helloRole = if (signalRole == Role.CALLER) "caller" else "answerer"
                        val hello = "{\"role\":\"$helloRole\",\"room_id\":\"${roomId.jsonEscape()}\",\"token\":\"${token.jsonEscape()}\"}\n"
                        socket.getOutputStream().write(hello.toByteArray(Charsets.UTF_8))
                        socket.getOutputStream().flush()
                        if (!isCurrentConnection(generation)) {
                            socket.close()
                            return@Thread
                        }
                        onLog("WebRTC relay connected: $host:$port room=$roomId")
                        startPeer(SignalChannel(socket, skipHandshakeAck = true), createOffer = signalRole == Role.CALLER, generation = generation)
                        return@Thread
                    } catch (e: Exception) {
                        lastError = e
                        onLog("WebRTC relay retry: ${e.message}")
                        Thread.sleep(300L)
                    }
                }
            } catch (e: Exception) {
                lastError = e
            }
            if (running.get()) onLog("WebRTC relay connect failed: ${lastError?.message}")
        }.start()
    }

    fun close() {
        onLog("WebRTC close on ${threadName()}")
        connectionGeneration.incrementAndGet()
        running.set(false)
        signal?.close()
        signal = null
        serverSocket?.close()
        serverSocket = null
        peerConnection?.close()
        peerConnection = null
        pendingRemoteCandidates.clear()
        localCapturer?.stopCapture()
        localCapturer?.dispose()
        localCapturer = null
        localSource?.dispose()
        localSource = null
        audioSource?.dispose()
        audioSource = null
        factory?.dispose()
        factory = null
        restoreAudioRoute()
    }

    fun release() {
        onLog("WebRTC release on ${threadName()}")
        close()
        runOnMainSync { renderer.release() }
        eglBase.release()
    }

    private fun beginConnection(): Int {
        close()
        val generation = connectionGeneration.incrementAndGet()
        running.set(true)
        return generation
    }

    private fun isCurrentConnection(generation: Int): Boolean =
        running.get() && connectionGeneration.get() == generation

    private fun startPeer(channel: SignalChannel, createOffer: Boolean, generation: Int) {
        if (!isCurrentConnection(generation)) {
            channel.close()
            return
        }
        onLog("WebRTC startPeer createOffer=$createOffer on ${threadName()}")
        configureSpeakerRoute()
        signal = channel
        ensureFactory()
        val servers = iceServers()
        onLog("WebRTC ICE servers: ${servers.joinToString { it.urls.joinToString("|") }}")
        peerConnection = factory?.createPeerConnection(servers, peerObserver())
        addLocalMedia()
        if (!channel.listen { message ->
                if (isCurrentConnection(generation)) handleSignal(message)
            }) {
            onLog("WebRTC signal listener failed to start")
            channel.close()
            return
        }
        if (createOffer) createOffer()
    }

    private fun ensureFactory() {
        if (factory != null) return
        onLog("WebRTC ensureFactory on ${threadName()}")
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(context.applicationContext)
                .createInitializationOptions()
        )
        val encoderFactory = DefaultVideoEncoderFactory(eglBase.eglBaseContext, true, true)
        val decoderFactory = DefaultVideoDecoderFactory(eglBase.eglBaseContext)
        factory = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(encoderFactory)
            .setVideoDecoderFactory(decoderFactory)
            .createPeerConnectionFactory()
    }

    private fun addLocalMedia() {
        onLog("WebRTC addLocalMedia on ${threadName()}")
        val factory = factory ?: return
        localCapturer = createCameraCapturer()
        if (localCapturer == null) {
            onLog("WebRTC camera capturer unavailable")
            return
        }
        localSource = factory.createVideoSource(false)
        val textureHelper = SurfaceTextureHelper.create("WebRtcCamera", eglBase.eglBaseContext)
        localCapturer?.initialize(textureHelper, context.applicationContext, localSource?.capturerObserver)
        try {
            localCapturer?.startCapture(320, 240, 15)
        } catch (e: Exception) {
            onLog("WebRTC camera start failed: ${e.message}")
            return
        }

        val videoTrack = factory.createVideoTrack("video0", localSource)
        peerConnection?.addTrack(videoTrack, listOf("cruisecar"))
        onLog("WebRTC local video track added")

        audioSource = factory.createAudioSource(MediaConstraints())
        val audioTrack: AudioTrack = factory.createAudioTrack("audio0", audioSource)
        audioTrack.setEnabled(true)
        peerConnection?.addTrack(audioTrack, listOf("cruisecar"))
        onLog("WebRTC local audio track added")
    }

    private fun configureSpeakerRoute() {
        runOnMainSync {
            val manager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            if (audioManager == null) {
                audioManager = manager
                previousAudioMode = manager.mode
                previousSpeakerphoneOn = manager.isSpeakerphoneOn
            }
            manager.mode = AudioManager.MODE_IN_COMMUNICATION
            manager.isSpeakerphoneOn = true
            onLog("WebRTC audio route speaker=${manager.isSpeakerphoneOn} mode=${manager.mode}")
        }
    }

    private fun restoreAudioRoute() {
        runOnMainSync {
            val manager = audioManager ?: return@runOnMainSync
            previousSpeakerphoneOn?.let { manager.isSpeakerphoneOn = it }
            previousAudioMode?.let { manager.mode = it }
            onLog("WebRTC audio route restored speaker=${manager.isSpeakerphoneOn} mode=${manager.mode}")
            audioManager = null
            previousAudioMode = null
            previousSpeakerphoneOn = null
        }
    }

    private fun createOffer() {
        peerConnection?.createOffer(object : SimpleSdpObserver() {
            override fun onCreateSuccess(desc: SessionDescription) {
                onLog("WebRTC offer created")
                peerConnection?.setLocalDescription(SimpleSdpObserver(), desc)
                signal?.sendSdp(desc)
            }
        }, MediaConstraints())
    }

    private fun createAnswer() {
        peerConnection?.createAnswer(object : SimpleSdpObserver() {
            override fun onCreateSuccess(desc: SessionDescription) {
                onLog("WebRTC answer created")
                peerConnection?.setLocalDescription(SimpleSdpObserver(), desc)
                signal?.sendSdp(desc)
            }
        }, MediaConstraints())
    }

    private fun handleSignal(message: JSONObject) {
        when (message.optString("type")) {
            "peer_replaced" -> {
                onLog("WebRTC peer replaced: ${message.optString("role")}")
                onPeerDisconnected()
            }
            "peer_left" -> {
                onLog("WebRTC peer left: ${message.optString("role")}")
                onPeerDisconnected()
            }
            "offer", "answer" -> {
                onLog("WebRTC received ${message.optString("type")}")
                val type = SessionDescription.Type.fromCanonicalForm(message.getString("type"))
                val sdp = SessionDescription(type, message.getString("sdp"))
                peerConnection?.setRemoteDescription(object : SimpleSdpObserver() {
                    override fun onSetSuccess() {
                        flushPendingRemoteCandidates()
                        if (type == SessionDescription.Type.OFFER) createAnswer()
                    }
                }, sdp)
            }
            "candidate" -> {
                val candidate = IceCandidate(
                    message.getString("sdpMid"),
                    message.getInt("sdpMLineIndex"),
                    message.getString("candidate")
                )
                onLog("WebRTC received ICE candidate: ${candidate.sdp.summarizeIceCandidate()}")
                if (peerConnection?.remoteDescription == null) {
                    pendingRemoteCandidates += candidate
                    onLog("WebRTC queued remote ICE until SDP is set")
                } else {
                    peerConnection?.addIceCandidate(candidate)
                }
            }
        }
    }

    private fun flushPendingRemoteCandidates() {
        val pc = peerConnection ?: return
        if (pendingRemoteCandidates.isEmpty()) return
        val queued = pendingRemoteCandidates.toList()
        pendingRemoteCandidates.clear()
        queued.forEach { pc.addIceCandidate(it) }
        onLog("WebRTC flushed ${queued.size} queued remote ICE candidates")
    }

    private fun peerObserver() = object : PeerConnection.Observer {
        override fun onIceCandidate(candidate: IceCandidate) {
            onLog("WebRTC sending ICE candidate: ${candidate.sdp.summarizeIceCandidate()}")
            signal?.sendCandidate(candidate)
        }

        override fun onTrack(transceiver: org.webrtc.RtpTransceiver) {
            transceiver.receiver.track()?.let { track ->
                when (track) {
                    is VideoTrack -> {
                        onLog("WebRTC remote video track received")
                        runOnMainSync { track.addSink(renderer) }
                    }
                    is AudioTrack -> {
                        onLog("WebRTC remote audio track received")
                        track.setEnabled(true)
                    }
                    else -> onLog("WebRTC remote track received: ${track.kind()}")
                }
            }
        }

        override fun onAddStream(stream: MediaStream) {
            stream.videoTracks.firstOrNull()?.let { track ->
                onLog("WebRTC remote stream received")
                runOnMainSync { track.addSink(renderer) }
            }
        }

        override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
            onLog("WebRTC ICE: $state")
        }

        override fun onSignalingChange(state: PeerConnection.SignalingState) = Unit
        override fun onIceConnectionReceivingChange(receiving: Boolean) = Unit
        override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
            onLog("WebRTC ICE gathering: $state")
        }
        override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) = Unit
        override fun onAddTrack(receiver: RtpReceiver, streams: Array<out MediaStream>) = Unit
        override fun onRemoveStream(stream: MediaStream) = Unit
        override fun onDataChannel(channel: DataChannel) = Unit
        override fun onRenegotiationNeeded() = Unit
    }

    private fun createCameraCapturer(): VideoCapturer? {
        return createCameraCapturer(Camera1Enumerator(false), "Camera1")
            ?: createCameraCapturer(Camera2Enumerator(context), "Camera2")
    }

    private fun createCameraCapturer(enumerator: CameraEnumerator, label: String): VideoCapturer? {
        val back = enumerator.deviceNames.firstOrNull { enumerator.isBackFacing(it) }
        val front = enumerator.deviceNames.firstOrNull { enumerator.isFrontFacing(it) }
        val any = enumerator.deviceNames.firstOrNull()
        val preferred = if (cameraFacing == CameraFacing.FRONT) front else back
        val fallback = if (cameraFacing == CameraFacing.FRONT) back else front
        for (name in listOfNotNull(preferred, fallback, any).distinct()) {
            val capturer = enumerator.createCapturer(name, null)
            if (capturer != null) {
                onLog("WebRTC using $label ${cameraFacing.name.lowercase()} camera: $name")
                return capturer
            }
        }
        onLog("WebRTC $label camera unavailable")
        return null
    }

    private fun initRenderer() {
        onLog("WebRTC initRenderer requested on ${threadName()}")
        runOnMainSync {
            onLog("WebRTC initRenderer executing on ${threadName()}")
            renderer.init(eglBase.eglBaseContext, null)
            renderer.setMirror(role == Role.CALLER)
            renderer.setEnableHardwareScaler(true)
        }
    }

    private fun runOnMainSync(action: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            action()
            return
        }
        val latch = CountDownLatch(1)
        var failure: Throwable? = null
        Handler(Looper.getMainLooper()).post {
            try {
                action()
            } catch (t: Throwable) {
                failure = t
            } finally {
                latch.countDown()
            }
        }
        latch.await()
        failure?.let { throw it }
    }

    private fun threadName(): String =
        "${Thread.currentThread().name}/${Thread.currentThread().id}"

    private fun iceServers(): List<PeerConnection.IceServer> =
        buildList {
            val remoteServers = try {
                iceServerProvider()
            } catch (e: Exception) {
                onLog("WebRTC load ICE servers failed: ${e.message}")
                emptyList()
            }
            val configs = remoteServers.ifEmpty {
                listOf(IceServerConfig(listOf("stun:stun.l.google.com:19302")))
            }
            for (server in configs) {
                if (server.urls.isEmpty()) continue
                val builder = PeerConnection.IceServer.builder(server.urls)
                if (server.username.isNotBlank()) builder.setUsername(server.username)
                if (server.credential.isNotBlank()) builder.setPassword(server.credential)
                add(builder.createIceServer())
            }
        }
}

private fun String.summarizeIceCandidate(): String {
    val type = Regex(" typ ([a-zA-Z0-9]+)").find(this)?.groupValues?.getOrNull(1) ?: "unknown"
    val protocol = Regex("candidate:[^ ]+ [0-9]+ ([a-zA-Z]+)").find(this)?.groupValues?.getOrNull(1) ?: "unknown"
    val relay = if (type == "relay") " TURN" else ""
    return "$type/$protocol$relay"
}

private class SignalChannel(private val socket: Socket, private val skipHandshakeAck: Boolean = false) {
    private val input = DataInputStream(socket.getInputStream())
    private val output = DataOutputStream(socket.getOutputStream())
    private val running = AtomicBoolean(true)

    fun listen(onMessage: (JSONObject) -> Unit): Boolean {
        val thread = Thread {
            if (skipHandshakeAck) {
                try {
                    val ack = StringBuilder()
                    while (true) {
                        val b = input.readByte().toInt()
                        if (b == '\n'.code) break
                        ack.append(b.toChar())
                    }
                } catch (_: Exception) {
                    running.set(false)
                }
            }
            while (running.get()) {
                try {
                    val length = input.readInt()
                    if (length <= 0 || length > 1024 * 1024) break
                    val bytes = ByteArray(length)
                    input.readFully(bytes)
                    onMessage(JSONObject(String(bytes, Charsets.UTF_8)))
                } catch (_: Exception) {
                    break
                }
            }
        }
        return try {
            thread.name = "webrtc-signal"
            thread.start()
            true
        } catch (_: OutOfMemoryError) {
            running.set(false)
            close()
            false
        } catch (_: RuntimeException) {
            running.set(false)
            close()
            false
        }
    }

    @Synchronized
    fun sendSdp(desc: SessionDescription) {
        send(JSONObject().apply {
            put("type", desc.type.canonicalForm())
            put("sdp", desc.description)
        })
    }

    @Synchronized
    fun sendCandidate(candidate: IceCandidate) {
        send(JSONObject().apply {
            put("type", "candidate")
            put("sdpMid", candidate.sdpMid)
            put("sdpMLineIndex", candidate.sdpMLineIndex)
            put("candidate", candidate.sdp)
        })
    }

    private fun send(json: JSONObject) {
        val bytes = json.toString().toByteArray(Charsets.UTF_8)
        output.writeInt(bytes.size)
        output.write(bytes)
        output.flush()
    }

    fun close() {
        running.set(false)
        socket.close()
    }
}

private fun String.jsonEscape(): String =
    replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

private open class SimpleSdpObserver : SdpObserver {
    override fun onCreateSuccess(desc: SessionDescription) = Unit
    override fun onSetSuccess() = Unit
    override fun onCreateFailure(error: String) = Unit
    override fun onSetFailure(error: String) = Unit
}
