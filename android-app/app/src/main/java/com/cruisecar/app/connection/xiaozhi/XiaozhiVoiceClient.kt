package com.cruisecar.app.connection.xiaozhi

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import com.theeasiestway.opus.Constants
import com.theeasiestway.opus.Opus
import com.cruisecar.app.data.remote.RemoteApi
import com.cruisecar.app.data.remote.XiaozhiBridgeEvent
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class XiaozhiVoiceClient(
    private val context: Context,
    private val managerBaseUrl: String,
    private val authToken: String,
    private val deviceId: String,
    private val deviceName: String,
    private val onLog: (String) -> Unit,
) {
    private val executor = Executors.newSingleThreadExecutor()
    private val poller: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val recording = AtomicBoolean(false)
    private var pollTask: ScheduledFuture<*>? = null
    private var lastSeq = 0L
    private var encoder: Opus? = null
    private var decoder: Opus? = null
    private var recorder: AudioRecord? = null
    private var player: AudioTrack? = null
    private var uplinkFrames = 0
    private var downlinkFrames = 0

    fun connect() {
        executor.execute {
            try {
                RemoteApi.bridgeConnect(managerBaseUrl, authToken, deviceId, deviceName)
                onLog("xiaozhi voice bridge connected: device=$deviceId")
                startPolling()
            } catch (e: Exception) {
                onLog("xiaozhi voice bridge connect failed: ${e.message}")
            }
        }
    }

    fun startRecording() {
        if (!recording.compareAndSet(false, true)) return
        executor.execute {
            var started = false
            try {
                RemoteApi.bridgeAudioStart(managerBaseUrl, authToken, deviceId)
                encoder = createOpusEncoder()
                recorder = createAudioRecord()
                recorder?.startRecording()
                started = true
                uplinkFrames = 0
                onLog("xiaozhi voice recording started: 16k mono opus (bundled libopus)")
                captureLoop()
            } catch (e: Exception) {
                onLog("xiaozhi voice recording failed: ${e.message}")
            } finally {
                recording.set(false)
                stopRecorder()
                stopEncoder()
                if (started) {
                    try {
                        RemoteApi.bridgeAudioStop(managerBaseUrl, authToken, deviceId)
                        onLog("xiaozhi voice recording stopped: frames=$uplinkFrames")
                    } catch (e: Exception) {
                        onLog("xiaozhi voice stop failed: ${e.message}")
                    }
                }
            }
        }
    }

    fun stopRecording() {
        recording.set(false)
    }

    fun disconnect() {
        recording.set(false)
        pollTask?.cancel(false)
        pollTask = null
        executor.execute {
            try {
                RemoteApi.bridgeDisconnect(managerBaseUrl, authToken, deviceId)
                onLog("xiaozhi voice bridge disconnected")
            } catch (e: Exception) {
                onLog("xiaozhi voice bridge disconnect failed: ${e.message}")
            } finally {
                stopDecoder()
                stopPlayer()
            }
        }
    }

    fun shutdown() {
        disconnect()
        executor.shutdown()
        poller.shutdownNow()
    }

    private fun startPolling() {
        if (pollTask != null) return
        pollTask = poller.scheduleAtFixedRate({
            try {
                val events = RemoteApi.bridgeEvents(managerBaseUrl, authToken, deviceId, lastSeq)
                for (event in events) {
                    lastSeq = maxOf(lastSeq, event.seq)
                    handleEvent(event)
                }
            } catch (e: Exception) {
                onLog("xiaozhi voice event poll failed: ${e.message}")
            }
        }, 0, 500, TimeUnit.MILLISECONDS)
    }

    private fun handleEvent(event: XiaozhiBridgeEvent) {
        when (event.type) {
            "connected" -> onLog("xiaozhi event connected: session=${event.sessionId.ifBlank { "-" }}")
            "stt" -> onLog("xiaozhi STT: ${event.text}")
            "llm" -> onLog("xiaozhi LLM: ${event.text}")
            "tts" -> onLog("xiaozhi TTS ${event.state}: ${event.text}")
            "audio_downlink" -> playOpus(event.audio)
            "audio_uplink" -> Unit
            else -> onLog("xiaozhi event ${event.type}: state=${event.state} size=${event.audioSize}")
        }
    }

    private fun captureLoop() {
        val record = recorder ?: return
        val codec = encoder ?: return
        val pcm = ByteArray(INPUT_PCM_BYTES)
        while (recording.get()) {
            val read = record.read(pcm, 0, pcm.size)
            if (read <= 0) continue
            if (read != pcm.size) continue
            val frame = codec.encode(pcm, Constants.FrameSize.Companion._960())
            if (frame != null && frame.isNotEmpty()) {
                try {
                    RemoteApi.bridgeAudioFrame(managerBaseUrl, authToken, deviceId, frame)
                    uplinkFrames += 1
                    if (uplinkFrames == 1 || uplinkFrames % 25 == 0) {
                        onLog("xiaozhi voice uplink opus frames=$uplinkFrames last=${frame.size}B")
                    }
                } catch (e: Exception) {
                    onLog("xiaozhi voice send frame failed: ${e.message}")
                }
            }
        }
    }

    private fun playOpus(frame: ByteArray) {
        if (frame.isEmpty()) return
        try {
            val codec = decoder ?: createOpusDecoder().also { decoder = it }
            val track = player ?: createAudioTrack().also {
                player = it
                it.play()
            }
            val pcm = codec.decode(frame, Constants.FrameSize.Companion._custom(OUTPUT_FRAME_SAMPLES))
            if (pcm != null && pcm.isNotEmpty()) {
                track.write(pcm, 0, pcm.size)
            }
            downlinkFrames += 1
            if (downlinkFrames == 1 || downlinkFrames % 25 == 0) {
                onLog("xiaozhi voice downlink opus frames=$downlinkFrames last=${frame.size}B")
            }
        } catch (e: Exception) {
            onLog("xiaozhi voice playback failed: ${e.message}")
            stopDecoder()
            stopPlayer()
        }
    }

    private fun createAudioRecord(): AudioRecord {
        val minBuffer = AudioRecord.getMinBufferSize(
            INPUT_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBuffer, INPUT_PCM_BYTES * 4)
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            INPUT_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            throw IllegalStateException("AudioRecord not initialized")
        }
        return record
    }

    private fun createOpusEncoder(): Opus {
        return Opus().apply {
            check(encoderInit(
                Constants.SampleRate.Companion._16000(),
                Constants.Channels.Companion.mono(),
                Constants.Application.Companion.voip()
            ) >= 0) { "libopus encoder init failed" }
            encoderSetBitrate(Constants.Bitrate.Companion.instance(24_000))
        }
    }

    private fun createOpusDecoder(): Opus {
        return Opus().apply {
            check(decoderInit(
                Constants.SampleRate.Companion._24000(),
                Constants.Channels.Companion.mono()
            ) >= 0) { "libopus decoder init failed" }
        }
    }

    private fun createAudioTrack(): AudioTrack {
        val minBuffer = AudioTrack.getMinBufferSize(
            OUTPUT_SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(OUTPUT_SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuffer, OUTPUT_PCM_BYTES * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    private fun stopRecorder() {
        try {
            recorder?.stop()
        } catch (_: Exception) {
        }
        recorder?.release()
        recorder = null
    }

    private fun stopEncoder() {
        try {
            encoder?.encoderRelease()
        } catch (_: Exception) {
        }
        encoder = null
    }

    private fun stopDecoder() {
        try {
            decoder?.decoderRelease()
        } catch (_: Exception) {
        }
        decoder = null
    }

    private fun stopPlayer() {
        try {
            player?.stop()
        } catch (_: Exception) {
        }
        player?.release()
        player = null
    }

    companion object {
        private const val CHANNELS = 1
        private const val INPUT_SAMPLE_RATE = 16_000
        private const val OUTPUT_SAMPLE_RATE = 24_000
        private const val INPUT_FRAME_SAMPLES = 960
        private const val INPUT_PCM_BYTES = INPUT_FRAME_SAMPLES * 2
        private const val OUTPUT_FRAME_SAMPLES = 1440
        private const val OUTPUT_PCM_BYTES = OUTPUT_FRAME_SAMPLES * 2
    }
}
