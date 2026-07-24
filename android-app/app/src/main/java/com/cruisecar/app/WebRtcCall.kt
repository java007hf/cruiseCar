package com.cruisecar.app

import android.content.Context
import org.json.JSONObject
import org.webrtc.AudioSource
import org.webrtc.AudioTrack
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
import java.util.concurrent.atomic.AtomicBoolean

class WebRtcCall(
    private val context: Context,
    private val role: Role,
    private val renderer: SurfaceViewRenderer,
    private val onLog: (String) -> Unit
) {
    enum class Role { CALLER, ANSWERER }

    private val eglBase = EglBase.create()
    private var factory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var signal: SignalChannel? = null
    private var serverSocket: ServerSocket? = null
    private var localCapturer: VideoCapturer? = null
    private var localSource: VideoSource? = null
    private var audioSource: AudioSource? = null
    private val running = AtomicBoolean(false)

    fun startServer(port: Int) {
        if (!running.compareAndSet(false, true)) return
        initRenderer()
        Thread {
            try {
                serverSocket = ServerSocket(port)
                onLog("WebRTC signaling server started on $port")
                val socket = serverSocket?.accept() ?: return@Thread
                onLog("WebRTC peer connected: ${socket.inetAddress.hostAddress}")
                startPeer(SignalChannel(socket), createOffer = true)
            } catch (e: Exception) {
                if (running.get()) onLog("WebRTC server failed: ${e.message}")
            }
        }.start()
    }

    fun connect(host: String, port: Int) {
        close()
        running.set(true)
        initRenderer()
        Thread {
            var lastError: Exception? = null
            try {
                repeat(12) { attempt ->
                    try {
                        val socket = Socket(host, port)
                        onLog("WebRTC signaling connected: $host:$port")
                        startPeer(SignalChannel(socket), createOffer = false)
                        return@Thread
                    } catch (e: Exception) {
                        lastError = e
                        Thread.sleep(250L + attempt * 50L)
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

    fun close() {
        running.set(false)
        signal?.close()
        signal = null
        serverSocket?.close()
        serverSocket = null
        peerConnection?.close()
        peerConnection = null
        localCapturer?.stopCapture()
        localCapturer?.dispose()
        localCapturer = null
        localSource?.dispose()
        localSource = null
        audioSource?.dispose()
        audioSource = null
        factory?.dispose()
        factory = null
    }

    fun release() {
        close()
        renderer.release()
        eglBase.release()
    }

    private fun startPeer(channel: SignalChannel, createOffer: Boolean) {
        signal = channel
        ensureFactory()
        peerConnection = factory?.createPeerConnection(iceServers(), peerObserver())
        if (role == Role.CALLER) addLocalMedia()
        channel.listen { message -> handleSignal(message) }
        if (createOffer) createOffer()
    }

    private fun ensureFactory() {
        if (factory != null) return
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
        val factory = factory ?: return
        localCapturer = createCameraCapturer() ?: return
        localSource = factory.createVideoSource(false)
        val textureHelper = SurfaceTextureHelper.create("WebRtcCamera", eglBase.eglBaseContext)
        localCapturer?.initialize(textureHelper, context.applicationContext, localSource?.capturerObserver)
        localCapturer?.startCapture(640, 480, 24)

        val videoTrack = factory.createVideoTrack("video0", localSource)
        videoTrack.addSink(renderer)
        peerConnection?.addTrack(videoTrack, listOf("cruisecar"))

        audioSource = factory.createAudioSource(MediaConstraints())
        val audioTrack: AudioTrack = factory.createAudioTrack("audio0", audioSource)
        peerConnection?.addTrack(audioTrack, listOf("cruisecar"))
    }

    private fun createOffer() {
        peerConnection?.createOffer(object : SimpleSdpObserver() {
            override fun onCreateSuccess(desc: SessionDescription) {
                peerConnection?.setLocalDescription(SimpleSdpObserver(), desc)
                signal?.sendSdp(desc)
            }
        }, MediaConstraints())
    }

    private fun createAnswer() {
        peerConnection?.createAnswer(object : SimpleSdpObserver() {
            override fun onCreateSuccess(desc: SessionDescription) {
                peerConnection?.setLocalDescription(SimpleSdpObserver(), desc)
                signal?.sendSdp(desc)
            }
        }, MediaConstraints())
    }

    private fun handleSignal(message: JSONObject) {
        when (message.optString("type")) {
            "offer", "answer" -> {
                val type = SessionDescription.Type.fromCanonicalForm(message.getString("type"))
                val sdp = SessionDescription(type, message.getString("sdp"))
                peerConnection?.setRemoteDescription(object : SimpleSdpObserver() {
                    override fun onSetSuccess() {
                        if (type == SessionDescription.Type.OFFER) createAnswer()
                    }
                }, sdp)
            }
            "candidate" -> {
                peerConnection?.addIceCandidate(
                    IceCandidate(
                        message.getString("sdpMid"),
                        message.getInt("sdpMLineIndex"),
                        message.getString("candidate")
                    )
                )
            }
        }
    }

    private fun peerObserver() = object : PeerConnection.Observer {
        override fun onIceCandidate(candidate: IceCandidate) {
            signal?.sendCandidate(candidate)
        }

        override fun onTrack(transceiver: org.webrtc.RtpTransceiver) {
            transceiver.receiver.track()?.let { track ->
                if (track is VideoTrack) track.addSink(renderer)
            }
        }

        override fun onAddStream(stream: MediaStream) {
            stream.videoTracks.firstOrNull()?.addSink(renderer)
        }

        override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
            onLog("WebRTC ICE: $state")
        }

        override fun onSignalingChange(state: PeerConnection.SignalingState) = Unit
        override fun onIceConnectionReceivingChange(receiving: Boolean) = Unit
        override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) = Unit
        override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) = Unit
        override fun onAddTrack(receiver: RtpReceiver, streams: Array<out MediaStream>) = Unit
        override fun onRemoveStream(stream: MediaStream) = Unit
        override fun onDataChannel(channel: DataChannel) = Unit
        override fun onRenegotiationNeeded() = Unit
    }

    private fun createCameraCapturer(): VideoCapturer? {
        val enumerator: CameraEnumerator = Camera2Enumerator(context)
        val front = enumerator.deviceNames.firstOrNull { enumerator.isFrontFacing(it) }
        val any = enumerator.deviceNames.firstOrNull()
        return (front ?: any)?.let { enumerator.createCapturer(it, null) }
    }

    private fun initRenderer() {
        renderer.init(eglBase.eglBaseContext, null)
        renderer.setMirror(role == Role.CALLER)
        renderer.setEnableHardwareScaler(true)
    }

    private fun iceServers(): List<PeerConnection.IceServer> =
        listOf(PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer())
}

private class SignalChannel(private val socket: Socket) {
    private val input = DataInputStream(socket.getInputStream())
    private val output = DataOutputStream(socket.getOutputStream())
    private val running = AtomicBoolean(true)

    fun listen(onMessage: (JSONObject) -> Unit) {
        Thread {
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
        }.start()
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

private open class SimpleSdpObserver : SdpObserver {
    override fun onCreateSuccess(desc: SessionDescription) = Unit
    override fun onSetSuccess() = Unit
    override fun onCreateFailure(error: String) = Unit
    override fun onSetFailure(error: String) = Unit
}
