package com.pigeonvision.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.pigeonvision.app.SearchResponse
import com.pigeonvision.app.SearchResult

@Composable
fun SearchScreen(
  image: Bitmap?,
  state: ActionState<SearchResponse>,
  onPickImage: () -> Unit,
  onCameraImage: () -> Unit,
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
        if (state.value.results.isEmpty()) {
          item { Text("未返回检索结果") }
        } else {
          items(state.value.results) { result -> SearchResultRow(result) }
        }
      }
    }
    item { Spacer(Modifier.height(12.dp)) }
  }
}

@Composable
private fun SearchResultRow(result: SearchResult) {
  Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
      Text("#${result.rank}", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
      Column(modifier = Modifier.weight(1f)) {
        Text(result.bloodName, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text(result.pgId, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
      }
      Text(result.distance.formatDistance(), fontWeight = FontWeight.Medium)
    }
    HorizontalDivider()
  }
}
