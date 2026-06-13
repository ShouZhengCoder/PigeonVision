package com.pigeonvision.app.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.pigeonvision.app.ApiClient
import com.pigeonvision.app.CompareResponse
import com.pigeonvision.app.EyeCrop
import com.pigeonvision.app.HealthResponse
import com.pigeonvision.app.SearchResponse
import com.pigeonvision.app.SearchImageLoader
import com.pigeonvision.app.YoloDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface ActionState<out T> {
  data object Idle : ActionState<Nothing>
  data object Loading : ActionState<Nothing>
  data class Success<T>(val value: T) : ActionState<T>
  data class Error(val message: String) : ActionState<Nothing>
}

private enum class ImageTarget {
  CompareA,
  CompareB,
  Search,
}

private class NoEyeDetectedException(message: String) : Exception(message)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PigeonVisionApp(
  apiClient: ApiClient,
  detector: YoloDetector,
  decodeBitmap: suspend (Uri) -> Bitmap,
  createCameraUri: () -> Uri,
  showToast: (String) -> Unit,
  modifier: Modifier = Modifier,
) {
  val context = LocalContext.current
  val prefs = remember { context.getSharedPreferences("pigeonvision", Context.MODE_PRIVATE) }
  var serverUrl by rememberSaveable { mutableStateOf(prefs.getString("server_url", "").orEmpty()) }
  var selectedTab by rememberSaveable { mutableIntStateOf(0) }
  var compareA by remember { mutableStateOf<Bitmap?>(null) }
  var compareB by remember { mutableStateOf<Bitmap?>(null) }
  var searchImage by remember { mutableStateOf<Bitmap?>(null) }
  var searchTopKText by rememberSaveable { mutableStateOf(ApiClient.DEFAULT_SEARCH_TOP_K.toString()) }
  var searchPage by rememberSaveable { mutableIntStateOf(0) }
  var imageTarget by remember { mutableStateOf<ImageTarget?>(null) }
  var cameraUri by remember { mutableStateOf<Uri?>(null) }
  var healthState by remember { mutableStateOf<ActionState<HealthResponse>>(ActionState.Idle) }
  var compareState by remember { mutableStateOf<ActionState<CompareResponse>>(ActionState.Idle) }
  var searchState by remember { mutableStateOf<ActionState<SearchResponse>>(ActionState.Idle) }
  var compareProcessPreview by remember { mutableStateOf<CompareProcessPreview?>(null) }
  var searchProcessPreview by remember { mutableStateOf<SearchProcessPreview?>(null) }
  val scope = rememberCoroutineScope()
  val searchImageLoader = remember { SearchImageLoader(apiClient) }

  LaunchedEffect(serverUrl) {
    prefs.edit().putString("server_url", serverUrl).apply()
  }

  fun setBitmap(target: ImageTarget, bitmap: Bitmap) {
    when (target) {
      ImageTarget.CompareA -> {
        compareA = bitmap
        compareState = ActionState.Idle
        compareProcessPreview = null
      }
      ImageTarget.CompareB -> {
        compareB = bitmap
        compareState = ActionState.Idle
        compareProcessPreview = null
      }
      ImageTarget.Search -> {
        searchImage = bitmap
        searchState = ActionState.Idle
        searchProcessPreview = null
        searchPage = 0
      }
    }
  }

  fun handleImageUri(uri: Uri?) {
    val target = imageTarget
    if (uri == null || target == null) return
    scope.launch {
      try {
        val bitmap = withContext(Dispatchers.IO) { decodeBitmap(uri) }
        setBitmap(target, bitmap)
      } catch (error: Throwable) {
        showToast(error.userMessage("图片读取失败"))
      }
    }
  }

  val galleryLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
      handleImageUri(uri)
    }

  val cameraLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
      if (success) {
        handleImageUri(cameraUri)
      }
    }

  fun launchCameraForCurrentTarget() {
    val target = imageTarget ?: return
    imageTarget = target
    val uri = createCameraUri()
    cameraUri = uri
    cameraLauncher.launch(uri)
  }

  val cameraPermissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) {
        launchCameraForCurrentTarget()
      } else {
        showToast("需要相机权限")
      }
    }

  fun chooseFromGallery(target: ImageTarget) {
    imageTarget = target
    galleryLauncher.launch("image/*")
  }

  fun takePhoto(target: ImageTarget) {
    imageTarget = target
    val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    if (granted) {
      launchCameraForCurrentTarget()
    } else {
      cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }
  }

  fun testConnection() {
    scope.launch {
      healthState = ActionState.Loading
      healthState =
        try {
          ActionState.Success(apiClient.health(serverUrl))
        } catch (error: Throwable) {
          ActionState.Error(error.userMessage("连接失败")).also { showToast(it.message) }
        }
    }
  }

  fun runCompare() {
    val imageA = compareA
    val imageB = compareB
    if (serverUrl.isBlank()) {
      showToast("请先输入服务器地址")
      return
    }
    if (imageA == null || imageB == null) {
      showToast("请选择两张图片")
      return
    }
    scope.launch {
      compareProcessPreview = null
      compareState = ActionState.Loading
      compareState =
        try {
          val cropA = cropOrThrow(detector, imageA, "第一张图片")
          compareProcessPreview = CompareProcessPreview(eyeCropA = cropA.bitmap, eyeCropB = null)
          val cropB = cropOrThrow(detector, imageB, "第二张图片")
          compareProcessPreview = CompareProcessPreview(eyeCropA = cropA.bitmap, eyeCropB = cropB.bitmap)
          val response =
            apiClient.compare(
              serverUrl,
              cropA.jpegBytes,
              cropB.jpegBytes,
              eyeCrop = true,
            )
          ActionState.Success(response)
        } catch (error: NoEyeDetectedException) {
          showToast("未检测到眼部，请靠近拍摄")
          ActionState.Error(error.message.orEmpty())
        } catch (error: Throwable) {
          ActionState.Error(error.userMessage("比对失败")).also { showToast(it.message) }
        }
    }
  }

  fun runSearch() {
    val image = searchImage
    val topK = searchTopKText.toIntOrNull()
    if (serverUrl.isBlank()) {
      showToast("请先输入服务器地址")
      return
    }
    if (image == null) {
      showToast("请选择一张图片")
      return
    }
    if (topK == null || topK !in 1..ApiClient.MAX_SEARCH_TOP_K) {
      showToast("Top-K 请输入 1 到 ${ApiClient.MAX_SEARCH_TOP_K}")
      return
    }
    scope.launch {
      searchProcessPreview = null
      searchState = ActionState.Loading
      searchPage = 0
      searchState =
        try {
          val crop = cropOrThrow(detector, image, "检索图片")
          searchProcessPreview = SearchProcessPreview(eyeCrop = crop.bitmap)
          val response =
            apiClient.search(serverUrl, crop.jpegBytes, topK = topK, eyeCrop = true)
          ActionState.Success(response)
        } catch (error: NoEyeDetectedException) {
          showToast("未检测到眼部，请靠近拍摄")
          ActionState.Error(error.message.orEmpty())
        } catch (error: Throwable) {
          ActionState.Error(error.userMessage("检索失败")).also { showToast(it.message) }
        }
    }
  }

  Scaffold(
    topBar = {
      TopAppBar(
        title = {
          Column {
            Text("PigeonVision", fontWeight = FontWeight.Bold)
            Text("鸽眼虹膜识别实验台", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
          }
        }
      )
    },
    modifier = modifier.fillMaxSize(),
  ) { padding ->
    Column(
      modifier =
        Modifier
          .padding(padding)
          .padding(horizontal = 16.dp)
          .fillMaxSize(),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      ServerPanel(
        serverUrl = serverUrl,
        onServerUrlChange = { serverUrl = it },
        healthState = healthState,
        onTestConnection = ::testConnection,
      )

      PrimaryTabRow(selectedTabIndex = selectedTab) {
        Tab(selected = selectedTab == 0, onClick = { selectedTab = 0 }, text = { Text("品种比对") })
        Tab(selected = selectedTab == 1, onClick = { selectedTab = 1 }, text = { Text("品种检索") })
      }

      when (selectedTab) {
        0 ->
          CompareScreen(
            imageA = compareA,
            imageB = compareB,
            state = compareState,
            processPreview = compareProcessPreview,
            onPickA = { chooseFromGallery(ImageTarget.CompareA) },
            onCameraA = { takePhoto(ImageTarget.CompareA) },
            onPickB = { chooseFromGallery(ImageTarget.CompareB) },
            onCameraB = { takePhoto(ImageTarget.CompareB) },
            onCompare = ::runCompare,
            modifier = Modifier.weight(1f),
          )
        else ->
          SearchScreen(
            image = searchImage,
            state = searchState,
            processPreview = searchProcessPreview,
            serverUrl = serverUrl,
            topKText = searchTopKText,
            pageIndex = searchPage,
            imageLoader = searchImageLoader,
            onPickImage = { chooseFromGallery(ImageTarget.Search) },
            onCameraImage = { takePhoto(ImageTarget.Search) },
            onTopKChange = { value ->
              if (value.length <= 3 && value.all(Char::isDigit)) {
                searchTopKText = value
                searchPage = 0
              }
            },
            onPageChange = { page -> searchPage = page.coerceAtLeast(0) },
            onSearch = ::runSearch,
            modifier = Modifier.weight(1f),
          )
      }
    }
  }
}

@Composable
private fun ServerPanel(
  serverUrl: String,
  onServerUrlChange: (String) -> Unit,
  healthState: ActionState<HealthResponse>,
  onTestConnection: () -> Unit,
) {
  val statusColor =
    when (healthState) {
      ActionState.Idle -> MaterialTheme.colorScheme.onSurfaceVariant
      ActionState.Loading -> MaterialTheme.colorScheme.secondary
      is ActionState.Success -> MaterialTheme.colorScheme.primary
      is ActionState.Error -> MaterialTheme.colorScheme.error
    }
  val statusText =
    when (healthState) {
      ActionState.Idle -> "未连接"
      ActionState.Loading -> "连接中"
      is ActionState.Success -> "在线"
      is ActionState.Error -> "异常"
    }

  OutlinedCard(
    shape = RoundedCornerShape(8.dp),
    colors = CardDefaults.outlinedCardColors(containerColor = MaterialTheme.colorScheme.surface),
  ) {
    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Column {
          Text("服务端连接", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
          Text("Flask / U-Net / FAISS", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Surface(shape = RoundedCornerShape(999.dp), color = statusColor.copy(alpha = 0.12f)) {
          Text(
            statusText,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            style = MaterialTheme.typography.labelSmall,
            color = statusColor,
            fontWeight = FontWeight.Bold,
          )
        }
      }

      Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
          value = serverUrl,
          onValueChange = onServerUrlChange,
          label = { Text("服务器地址") },
          placeholder = { Text("http://192.168.1.x:8080") },
          singleLine = true,
          modifier = Modifier.weight(1f),
        )
        Spacer(Modifier.width(8.dp))
        Button(onClick = onTestConnection, enabled = healthState !is ActionState.Loading, modifier = Modifier.height(56.dp)) {
          if (healthState is ActionState.Loading) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
          } else {
            Icon(Icons.Filled.CheckCircle, contentDescription = null)
          }
          Spacer(Modifier.width(6.dp))
          Text("测试")
        }
      }

      when (healthState) {
        ActionState.Idle -> Text("输入局域网 Flask 服务地址后测试连接", style = MaterialTheme.typography.bodySmall)
        ActionState.Loading -> Text("正在连接服务器...", style = MaterialTheme.typography.bodySmall)
        is ActionState.Success ->
          Text(
            "服务状态：${healthState.value.status}，图库：${healthState.value.gallerySize}",
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
          )
        is ActionState.Error ->
          Text(healthState.message, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
      }
    }
  }
}

private suspend fun cropOrThrow(detector: YoloDetector, bitmap: Bitmap, label: String): EyeCrop =
  withContext(Dispatchers.Default) {
    detector.detectAndCropJpeg(bitmap) ?: throw NoEyeDetectedException("$label 未检测到眼部，请靠近拍摄")
  }

private fun Throwable.userMessage(fallback: String): String = message?.takeIf { it.isNotBlank() } ?: fallback
