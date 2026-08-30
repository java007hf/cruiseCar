package com.cruisecar.app.data.remote

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

data class RemoteReceiver(
    val deviceId: String,
    val name: String,
    val online: Boolean,
    val espConnected: Boolean,
    val mode: String
)

data class RemoteIceServer(
    val urls: List<String>,
    val username: String = "",
    val credential: String = ""
)

data class XiaozhiBridgeEvent(
    val seq: Long,
    val type: String,
    val sessionId: String,
    val state: String,
    val text: String,
    val audio: ByteArray,
    val audioSize: Int
)

object RemoteApi {
    fun login(baseUrl: String, username: String, password: String): String {
        val json = request(
            baseUrl,
            "/api/auth/login",
            "POST",
            JSONObject().put("username", username).put("password", password),
            token = null
        )
        return json.getJSONObject("data").getString("token")
    }

    fun addReceiver(baseUrl: String, authToken: String, deviceId: String, name: String) {
        request(
            baseUrl,
            "/api/receivers",
            "POST",
            JSONObject().put("device_id", deviceId).put("name", name).put("token", authToken),
            token = authToken
        )
    }

    fun listReceivers(baseUrl: String, authToken: String): List<RemoteReceiver> {
        val json = request(baseUrl, "/api/receivers", "GET", null, authToken)
        val arr = json.getJSONArray("data")
        return (0 until arr.length()).map { i -> arr.getJSONObject(i).toRemoteReceiver() }
    }

    fun deleteReceiver(baseUrl: String, authToken: String, deviceId: String) {
        request(baseUrl, "/api/receivers/${deviceId.urlEncode()}", "DELETE", null, authToken)
    }

    fun iceServers(baseUrl: String, authToken: String): List<RemoteIceServer> {
        val json = request(baseUrl, "/api/webrtc/ice-servers", "GET", null, authToken)
        val arr = json.getJSONObject("data").getJSONArray("ice_servers")
        return (0 until arr.length()).mapNotNull { i ->
            val item = arr.getJSONObject(i)
            val urls = when (val raw = item.opt("urls")) {
                is JSONArray -> (0 until raw.length()).map { idx -> raw.getString(idx) }
                is String -> listOf(raw)
                else -> emptyList()
            }
            if (urls.isEmpty()) {
                null
            } else {
                RemoteIceServer(
                    urls = urls,
                    username = item.optString("username", ""),
                    credential = item.optString("credential", "")
                )
            }
        }
    }

    fun bridgeConnect(baseUrl: String, authToken: String, deviceId: String, deviceName: String) {
        request(
            baseUrl,
            "/api/bridge/connect",
            "POST",
            JSONObject().put("device_id", deviceId).put("device_name", deviceName),
            authToken
        )
    }

    fun bridgeDisconnect(baseUrl: String, authToken: String, deviceId: String) {
        request(
            baseUrl,
            "/api/bridge/disconnect",
            "POST",
            JSONObject().put("device_id", deviceId),
            authToken
        )
    }

    fun bridgeAudioStart(baseUrl: String, authToken: String, deviceId: String) {
        request(
            baseUrl,
            "/api/bridge/audio/start",
            "POST",
            JSONObject().put("device_id", deviceId),
            authToken
        )
    }

    fun bridgeAudioFrame(baseUrl: String, authToken: String, deviceId: String, opusFrame: ByteArray) {
        request(
            baseUrl,
            "/api/bridge/audio/frame",
            "POST",
            JSONObject()
                .put("device_id", deviceId)
                .put("audio_b64", Base64.encodeToString(opusFrame, Base64.NO_WRAP)),
            authToken
        )
    }

    fun bridgeAudioStop(baseUrl: String, authToken: String, deviceId: String) {
        request(
            baseUrl,
            "/api/bridge/audio/stop",
            "POST",
            JSONObject().put("device_id", deviceId),
            authToken
        )
    }

    fun bridgeEvents(baseUrl: String, authToken: String, deviceId: String, afterSeq: Long): List<XiaozhiBridgeEvent> {
        val json = request(
            baseUrl,
            "/api/bridge/events?device_id=${deviceId.urlEncode()}&after_seq=$afterSeq&limit=100",
            "GET",
            null,
            authToken
        )
        val arr = json.getJSONArray("data")
        return (0 until arr.length()).map { i ->
            val item = arr.getJSONObject(i)
            val audioB64 = item.optString("audio_b64", "")
            XiaozhiBridgeEvent(
                seq = item.optLong("seq", 0L),
                type = item.optString("type", ""),
                sessionId = item.optString("session_id", ""),
                state = item.optString("state", ""),
                text = item.optString("text", ""),
                audio = if (audioB64.isBlank()) ByteArray(0) else Base64.decode(audioB64, Base64.DEFAULT),
                audioSize = item.optInt("audio_size", 0)
            )
        }
    }

    private fun request(baseUrl: String, path: String, method: String, body: JSONObject?, token: String?): JSONObject {
        val url = URL(baseUrl.trimEnd('/') + path)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 15000
            readTimeout = 15000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            token?.takeIf { it.isNotBlank() }?.let { setRequestProperty("Authorization", "Bearer $it") }
            if (body != null) {
                doOutput = true
                outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            }
        }
        val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
        val text = BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { it.readText() }
        val json = JSONObject(text)
        if (!json.optBoolean("ok", false)) throw IllegalStateException(json.optString("error", "request failed"))
        return json
    }

    private fun JSONObject.toRemoteReceiver(): RemoteReceiver = RemoteReceiver(
        deviceId = getString("device_id"),
        name = optString("name", ""),
        online = optBoolean("online", false),
        espConnected = optBoolean("esp_connected", false),
        mode = optString("mode", "manual")
    )

    private fun String.urlEncode(): String = java.net.URLEncoder.encode(this, Charsets.UTF_8.name())
}
