package com.cruisecar.app.data.remote

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

    private fun request(baseUrl: String, path: String, method: String, body: JSONObject?, token: String?): JSONObject {
        val url = URL(baseUrl.trimEnd('/') + path)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 5000
            readTimeout = 5000
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
