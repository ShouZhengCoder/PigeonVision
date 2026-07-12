package com.pigeonvision.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.pigeonvision.app.ApiClient
import com.pigeonvision.app.SearchResponse
import com.pigeonvision.app.SearchResult
import com.pigeonvision.app.SearchImageLoader

private const val SearchPageSize = 10

data class SearchProcessPreview(
  val eyeCrop: Bitmap?,
)

@Composable
fun SearchScreen(
  image: Bitmap?,
  state: ActionState<SearchResponse>,
  processPreview: SearchProcessPreview?,
  serverUrl: String,
  topKText: String,
  pageIndex: Int,
  imageLoader: SearchImageLoader,
  onPickImage: () -> Unit,
  onCameraImage: () -> Unit,
  onTopKChange: (String) -> Unit,
  onPageChange: (Int) -> Unit,
  onSearch: () -> Unit,
  modifier: Modifier = Modifier,
) {
  LazyColumn(
    modifier = modifier,
    verticalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    item {
      ResearchSectionHeader(
        title = "查询样本",
        subtitle = "客户端提取眼部裁剪图，服务端完成虹膜归一化与向量检索。",
      )
    }
    item {
      BoxWithConstraints {
        if (maxWidth >= 720.dp) {
          Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            ImagePickerSlot("查询图片", image, onPickImage = onPickImage, onTakePhoto = onCameraImage, modifier = Modifier.weight(1f))
            SearchControlCard(
              topKText = topKText,
              state = state,
              onTopKChange = onTopKChange,
              onSearch = onSearch,
              modifier = Modifier.weight(1f),
            )
          }
        } else {
          Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            ImagePickerSlot("查询图片", image, onPickImage = onPickImage, onTakePhoto = onCameraImage)
            SearchControlCard(topKText = topKText, state = state, onTopKChange = onTopKChange, onSearch = onSearch)
          }
        }
      }
    }
    when (state) {
      ActionState.Idle ->
        item {
          ResearchMessageCard(
            title = "等待检索",
            message = "结果、分页和查询图处理链路会在检索完成后显示。",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
        }
      ActionState.Loading ->
        item {
          ResearchMessageCard(
            title = "正在检索",
            message = "正在检测眼部、提交服务端并查询相似鸽眼图库。",
            color = MaterialTheme.colorScheme.secondary,
            loading = true,
          )
        }
      is ActionState.Error ->
        item {
          Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            ResearchMessageCard(
              title = "检索失败",
              message = state.message,
              color = MaterialTheme.colorScheme.error,
            )
            processPreview?.let { SearchPreviewProcessCard(image = image, preview = it) }
          }
        }
      is ActionState.Success -> {
        val response = state.value
        item { SearchProcessCard(image = image, response = response) }
        val results = response.results
        if (results.isEmpty()) {
          item {
            ResearchMessageCard(
              title = "未返回结果",
              message = "服务端没有返回相似样本。",
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
        } else {
          val totalPages = ((results.size + SearchPageSize - 1) / SearchPageSize).coerceAtLeast(1)
          val currentPage = pageIndex.coerceIn(0, totalPages - 1)
          val pageResults = results.drop(currentPage * SearchPageSize).take(SearchPageSize)
          item {
            ResearchSectionHeader(
              title = "检索结果",
              subtitle = "返回 ${results.size} 条相似记录，每页 $SearchPageSize 条。",
            )
          }
          item {
            PaginationControls(
              pageIndex = currentPage,
              totalPages = totalPages,
              onPageChange = onPageChange,
            )
          }
          items(pageResults, key = { it.rank }) { result ->
            SearchResultRow(result = result, serverUrl = serverUrl, imageLoader = imageLoader)
          }
        }
      }
    }
    item { Spacer(Modifier.height(12.dp)) }
  }
}

@Composable
private fun SearchPreviewProcessCard(image: Bitmap?, preview: SearchProcessPreview) {
  ProcessFlowCard(
    title = "🔬 已完成处理过程",
    badgeText = "CLIENT CROP",
    rows =
      listOf(
        listOf(
          ProcessStep("原图", image),
          ProcessStep("眼部裁剪", preview.eyeCrop),
        )
      ),
  )
}

@Composable
private fun SearchControlCard(
  topKText: String,
  state: ActionState<SearchResponse>,
  onTopKChange: (String) -> Unit,
  onSearch: () -> Unit,
  modifier: Modifier = Modifier,
) {
  OutlinedCard(
    modifier = modifier.fillMaxWidth(),
    shape = RoundedCornerShape(8.dp),
    colors = CardDefaults.outlinedCardColors(containerColor = MaterialTheme.colorScheme.surface),
  ) {
    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
      Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text("检索参数", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
        Text("Top-K 范围 1-100，默认 20", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
      }
      OutlinedTextField(
        value = topKText,
        onValueChange = onTopKChange,
        label = { Text("Top-K") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        supportingText = { Text("返回相似记录数量") },
        modifier = Modifier.fillMaxWidth(),
      )
      Button(
        onClick = onSearch,
        enabled = state !is ActionState.Loading,
        modifier = Modifier.fillMaxWidth().height(50.dp),
      ) {
        if (state is ActionState.Loading) {
          CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
        } else {
          Icon(Icons.Filled.Search, contentDescription = null)
        }
        Spacer(Modifier.width(8.dp))
        Text("开始检索")
      }
    }
  }
}

@Composable
private fun SearchProcessCard(image: Bitmap?, response: SearchResponse) {
  val eyeCrop = remember(response.eyeCrop) { ApiClient.base64ToBitmap(response.eyeCrop) }
  val irisRegion = remember(response.irisRegion) { ApiClient.base64ToBitmap(response.irisRegion) }
  val normalized = remember(response.normalized) { ApiClient.base64ToBitmap(response.normalized) }

  ProcessFlowCard(
    title = "🔬 查询图处理过程",
    badgeText = if (response.fallback == "server_yolo") "SERVER YOLO" else "PIPELINE",
    rows =
      listOf(
        listOf(
          ProcessStep("原图", image),
          ProcessStep("眼部裁剪", eyeCrop),
          ProcessStep("虹膜区域", irisRegion),
          ProcessStep("虹膜归一化", normalized, wide = true),
        )
      ),
  )
}

@Composable
private fun PaginationControls(pageIndex: Int, totalPages: Int, onPageChange: (Int) -> Unit) {
  if (totalPages <= 1) return
  Row(
    modifier = Modifier.fillMaxWidth(),
    horizontalArrangement = Arrangement.Center,
    verticalAlignment = Alignment.CenterVertically,
  ) {
    IconButton(onClick = { onPageChange(pageIndex - 1) }, enabled = pageIndex > 0) {
      Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, contentDescription = "上一页")
    }
    Text("${pageIndex + 1} / $totalPages", fontWeight = FontWeight.Medium)
    IconButton(onClick = { onPageChange(pageIndex + 1) }, enabled = pageIndex < totalPages - 1) {
      Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = "下一页")
    }
  }
}

@Composable
private fun SearchResultRow(result: SearchResult, serverUrl: String, imageLoader: SearchImageLoader) {
  OutlinedCard(
    modifier = Modifier.fillMaxWidth(),
    shape = RoundedCornerShape(8.dp),
    colors = CardDefaults.outlinedCardColors(containerColor = MaterialTheme.colorScheme.surface),
  ) {
    Row(
      modifier = Modifier.fillMaxWidth().padding(10.dp),
      horizontalArrangement = Arrangement.spacedBy(12.dp),
      verticalAlignment = Alignment.CenterVertically,
    ) {
      ResultThumbnail(result = result, serverUrl = serverUrl, imageLoader = imageLoader)
      Column(modifier = Modifier.weight(1f)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
          Surface(shape = RoundedCornerShape(999.dp), color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)) {
            Text(
              "#${result.rank}",
              modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.primary,
              fontWeight = FontWeight.Bold,
            )
          }
          Text(result.bloodName, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        val sub = listOf(result.pgId, result.bloodId, result.imgId).filter { it.isNotBlank() && it != "-" }.joinToString(" · ")
        if (sub.isNotBlank()) {
          Text(sub, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
      }
      Column(horizontalAlignment = Alignment.End) {
        Text("距离", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(result.distance.formatDistance(), fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.tertiary)
      }
    }
  }
}

@Composable
private fun ResultThumbnail(result: SearchResult, serverUrl: String, imageLoader: SearchImageLoader) {
  var bitmap by remember(result.imageUrl) { mutableStateOf<Bitmap?>(null) }
  var failed by remember(result.imageUrl) { mutableStateOf(false) }

  LaunchedEffect(serverUrl, result.imageUrl) {
    bitmap = null
    failed = false
    if (result.imageUrl.isBlank()) {
      failed = true
      return@LaunchedEffect
    }
    try {
      bitmap = imageLoader.load(serverUrl, result.imageUrl)
    } catch (_: Throwable) {
      failed = true
    }
  }

  Box(
    modifier =
      Modifier
        .width(82.dp)
        .aspectRatio(4f / 3f)
        .clip(RoundedCornerShape(8.dp))
        .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(8.dp))
        .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)),
    contentAlignment = Alignment.Center,
  ) {
    val loaded = bitmap
    when {
      loaded != null ->
        Image(
          bitmap = loaded.asImageBitmap(),
          contentDescription = result.imgId.ifBlank { "检索结果图片" },
          modifier = Modifier.matchParentSize(),
          contentScale = ContentScale.Crop,
        )
      failed -> Text("无图", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
      else -> CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
    }
  }
}
