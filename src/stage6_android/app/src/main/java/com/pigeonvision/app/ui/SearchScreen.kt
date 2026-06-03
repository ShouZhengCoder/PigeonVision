package com.pigeonvision.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.foundation.text.KeyboardOptions
import com.pigeonvision.app.SearchResponse
import com.pigeonvision.app.SearchResult
import com.pigeonvision.app.SearchImageLoader

private const val SearchPageSize = 10

@Composable
fun SearchScreen(
  image: Bitmap?,
  state: ActionState<SearchResponse>,
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
      ImagePickerSlot("检索图片", image, onPickImage = onPickImage, onTakePhoto = onCameraImage)
    }
    item {
      OutlinedTextField(
        value = topKText,
        onValueChange = onTopKChange,
        label = { Text("Top-K") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        supportingText = { Text("1-100") },
        modifier = Modifier.fillMaxWidth(),
      )
    }
    item {
      Button(
        onClick = onSearch,
        enabled = state !is ActionState.Loading,
        modifier = Modifier.fillMaxWidth(),
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
    when (state) {
      ActionState.Idle -> item { Text("结果将在检索完成后显示", style = MaterialTheme.typography.bodyMedium) }
      ActionState.Loading ->
        item {
          Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
            Text("正在检测眼部并检索...")
          }
        }
      is ActionState.Error -> item { Text(state.message, color = MaterialTheme.colorScheme.error) }
      is ActionState.Success -> {
        val results = state.value.results
        if (results.isEmpty()) {
          item { Text("未返回检索结果") }
        } else {
          val totalPages = ((results.size + SearchPageSize - 1) / SearchPageSize).coerceAtLeast(1)
          val currentPage = pageIndex.coerceIn(0, totalPages - 1)
          val pageResults = results.drop(currentPage * SearchPageSize).take(SearchPageSize)
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
  Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
      ResultThumbnail(result = result, serverUrl = serverUrl, imageLoader = imageLoader)
      Column(modifier = Modifier.weight(1f)) {
        Text("#${result.rank}  ${result.bloodName}", fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
        val sub = listOf(result.pgId, result.bloodId, result.imgId).filter { it.isNotBlank() && it != "-" }.joinToString(" · ")
        if (sub.isNotBlank()) {
          Text(sub, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
      }
      Text(result.distance.formatDistance(), fontWeight = FontWeight.Medium)
    }
    HorizontalDivider()
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
