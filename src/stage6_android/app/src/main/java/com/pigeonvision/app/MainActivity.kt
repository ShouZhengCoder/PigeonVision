package com.pigeonvision.app

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.FileProvider
import com.pigeonvision.app.theme.PigeonVisionTheme
import com.pigeonvision.app.ui.PigeonVisionApp
import java.io.File

class MainActivity : ComponentActivity() {
  private val apiClient = ApiClient()
  private val detector by lazy { YoloDetector(applicationContext) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    setContent {
      PigeonVisionTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
          PigeonVisionApp(
            apiClient = apiClient,
            detector = detector,
            decodeBitmap = { uri -> decodeBitmapFromUri(this, uri) },
            createCameraUri = ::createCameraImageUri,
            showToast = { message -> Toast.makeText(this, message, Toast.LENGTH_SHORT).show() },
          )
        }
      }
    }
  }

  private fun createCameraImageUri(): Uri {
    val cameraDir = File(cacheDir, "camera").apply { mkdirs() }
    val imageFile = File.createTempFile("pigeon_", ".jpg", cameraDir)
    return FileProvider.getUriForFile(this, "$packageName.fileprovider", imageFile)
  }
}
