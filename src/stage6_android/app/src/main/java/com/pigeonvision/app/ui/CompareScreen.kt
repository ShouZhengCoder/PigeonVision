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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.CompareArrows
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.pigeonvision.app.CompareResponse
import java.util.Locale

@Composable
fun CompareScreen(
  imageA: Bitmap?,
  imageB: Bitmap?,
  state: ActionState<CompareResponse>,
  onPickA: () -> Unit,
  onCameraA: () -> Unit,
  onPickB: () -> Unit,
  onCameraB: () -> Unit,
  onCompare: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Column(
    modifier = modifier.verticalScroll(rememberScrollState()),
    verticalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    ImagePickerSlot("图片 A", imageA, onPickImage = onPickA, onTakePhoto = onCameraA)
    ImagePickerSlot("图片 B", imageB, onPickImage = onPickB, onTakePhoto = onCameraB)
    Button(
      onClick = onCompare,
      enabled = state !is ActionState.Loading,
      modifier = Modifier.fillMaxWidth(),
    ) {
      if (state is ActionState.Loading) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
      } else {
        Icon(Icons.AutoMirrored.Filled.CompareArrows, contentDescription = null)
      }
      Spacer(Modifier.width(8.dp))
      Text("开始比对")
    }
    CompareResult(state)
    Spacer(Modifier.height(12.dp))
  }
}

@Composable
private fun CompareResult(state: ActionState<CompareResponse>) {
  when (state) {
    ActionState.Idle -> Text("结果将在比对完成后显示", style = MaterialTheme.typography.bodyMedium)
    ActionState.Loading ->
      Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
        Text("正在检测眼部并提交服务端...")
      }
    is ActionState.Error -> Text(state.message, color = MaterialTheme.colorScheme.error)
    is ActionState.Success -> {
      val result = state.value
      val judgment = if (result.sameFamily) "同一品种" else "不同品种"
      val judgmentColor = if (result.sameFamily) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
      Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(judgment, color = judgmentColor, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
        Text("欧氏距离：${result.distance.formatDistance()}")
        if (!result.threshold.isNaN()) {
          Text("阈值：${result.threshold.formatDistance()}", style = MaterialTheme.typography.bodySmall)
        }
      }
    }
  }
}

internal fun Double.formatDistance(): String = String.format(Locale.US, "%.4f", this)
