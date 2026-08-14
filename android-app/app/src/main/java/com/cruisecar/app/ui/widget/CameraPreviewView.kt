package com.cruisecar.app.ui.widget

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.view.Surface
import android.view.TextureView

class CameraPreviewView(context: Context) : TextureView(context) {
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var shouldRun = false
    private var opening = false
    private var sessionVersion = 0

    init {
        surfaceTextureListener = object : SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                start()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                stop()
                return true
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        shouldRun = true
        if (!isAvailable || cameraDevice != null || opening) return
        if (context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return

        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = manager.cameraIdList.firstOrNull() ?: return
        val version = ++sessionVersion
        opening = true
        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                opening = false
                if (!shouldRun || version != sessionVersion || !isAvailable) {
                    camera.close()
                    return
                }
                cameraDevice = camera
                startPreview(camera, version)
            }

            override fun onDisconnected(camera: CameraDevice) {
                opening = false
                camera.close()
                if (cameraDevice == camera) cameraDevice = null
            }

            override fun onError(camera: CameraDevice, error: Int) {
                opening = false
                camera.close()
                if (cameraDevice == camera) cameraDevice = null
            }
        }, null)
    }

    fun stop() {
        shouldRun = false
        opening = false
        sessionVersion++
        try {
            captureSession?.stopRepeating()
        } catch (_: Exception) {
        }
        try {
            captureSession?.close()
        } catch (_: Exception) {
        }
        captureSession = null
        try {
            cameraDevice?.close()
        } catch (_: Exception) {
        }
        cameraDevice = null
    }

    fun snapshot(width: Int = 320, height: Int = 240): Bitmap? =
        if (isAvailable) getBitmap(width, height) else null

    private fun startPreview(camera: CameraDevice, version: Int) {
        val texture = surfaceTexture ?: return
        if (!shouldRun || version != sessionVersion || cameraDevice != camera) return
        texture.setDefaultBufferSize(width.coerceAtLeast(640), height.coerceAtLeast(480))
        val surface = Surface(texture)
        val request = try {
            camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            }
        } catch (_: IllegalStateException) {
            return
        }

        try {
            camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    if (!shouldRun || version != sessionVersion || cameraDevice != camera) {
                        session.close()
                        return
                    }
                    captureSession = session
                    try {
                        session.setRepeatingRequest(request.build(), null, null)
                    } catch (_: IllegalStateException) {
                        if (captureSession == session) captureSession = null
                        session.close()
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    session.close()
                }
            }, null)
        } catch (_: IllegalStateException) {
        }
    }
}
