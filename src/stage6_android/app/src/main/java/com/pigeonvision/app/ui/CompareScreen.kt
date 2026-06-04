package com.pigeonvision.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.CompareArrows
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.pigeonvision.app.ApiClient
import com.pigeonvision.app.CompareResponse
import java.util.Locale

data class CompareProcessPreview(
  val eyeCropA: Bitmap?,
  val eyeCropB: Bitmap?,
)

@Composable
fun CompareScreen(
  imageA: Bitmap?,
  imageB: Bitmap?,
  state: ActionState<CompareResponse>,
  processPreview: CompareProcessPreview?,
  onPickA: () -> Unit,
  onCameraA: () -> Unit,
  onPickB: () -> Unit,
  onCameraB: () -> Unit,
  onCompare: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Column(
    modifier = modifier.verticalScroll(rememberScrollState()),
    verticalArrangement = Arrangement.spacedBy(14.dp),
  ) {
    ResearchSectionHeader(
      title = "输入样本",
      subtitle = "选择两张鸽眼原图，客户端先完成眼部检测和裁剪。",
    )
    BoxWithConstraints {
      if (maxWidth >= 720.dp) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
          ImagePickerSlot("样本 A", imageA, onPickImage = onPickA, onTakePhoto = onCameraA, modifier = Modifier.weight(1f))
          ImagePickerSlot("样本 B", imageB, onPickImage = onPickB, onTakePhoto = onCameraB, modifier = Modifier.weight(1f))
        }
      } else {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
          ImagePickerSlot("样本 A", imageA, onPickImage = onPickA, onTakePhoto = onCameraA)
          ImagePickerSlot("样本 B", imageB, onPickImage = onPickB, onTakePhoto = onCameraB)
        }
      }
    }
    Button(
      onClick = onCompare,
      enabled = state !is ActionState.Loading,
      modifier = Modifier.fillMaxWidth().height(50.dp),
    ) {
      if (state is ActionState.Loading) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
      } else {
        Icon(Icons.AutoMirrored.Filled.CompareArrows, contentDescription = null)
      }
      Spacer(Modifier.width(8.dp))
      Text("开始比对")
    }
    CompareResult(state = state, imageA = imageA, imageB = imageB, processPreview = processPreview)
    Spacer(Modifier.height(12.dp))
  }
}

@Composable
private fun CompareResult(
  state: ActionState<CompareResponse>,
  imageA: Bitmap?,
  imageB: Bitmap?,
  processPreview: CompareProcessPreview?,
) {
  when (state) {
    ActionState.Idle ->
      ResearchMessageCard(
        title = "等待比对",
        message = "结果和处理链路会在实验完成后显示。",
        color = MaterialTheme.colorScheme.onSurfaceVariant,
      )
    ActionState.Loading ->
      Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ResearchMessageCard(
          title = "正在运行比对",
          message = "正在检测眼部、提交服务端并等待虹膜特征距离。",
          color = MaterialTheme.colorScheme.secondary,
          loading = true,
        )
        processPreview?.let { ComparePreviewProcessCard(imageA = imageA, imageB = imageB, preview = it) }
      }
    is ActionState.Error ->
      Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ResearchMessageCard(
          title = "比对失败",
          message = state.message,
          color = MaterialTheme.colorScheme.error,
        )
        processPreview?.let { ComparePreviewProcessCard(imageA = imageA, imageB = imageB, preview = it) }
      }
    is ActionState.Success -> {
      val result = state.value
      val judgment = if (result.sameFamily) "同一品种" else "不同品种"
      val judgmentColor = if (result.sameFamily) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
      val eyeCropA = remember(result.eyeCropA) { ApiClient.base64ToBitmap(result.eyeCropA) }
      val irisRegionA = remember(result.irisRegionA) { ApiClient.base64ToBitmap(result.irisRegionA) }
      val normalizedA = remember(result.normalizedA) { ApiClient.base64ToBitmap(result.normalizedA) }
      val eyeCropB = remember(result.eyeCropB) { ApiClient.base64ToBitmap(result.eyeCropB) }
      val irisRegionB = remember(result.irisRegionB) { ApiClient.base64ToBitmap(result.irisRegionB) }
      val normalizedB = remember(result.normalizedB) { ApiClient.base64ToBitmap(result.normalizedB) }

      Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedCard(
          shape = RoundedCornerShape(8.dp),
          colors = CardDefaults.outlinedCardColors(containerColor = MaterialTheme.colorScheme.surface),
        ) {
          Column(modifier = Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.Start) {
              Text("实验结果", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
              Text(judgment, color = judgmentColor, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
              MetricTile("欧氏距离", result.distance.formatDistance(), MaterialTheme.colorScheme.primary, modifier = Modifier.weight(1f))
              MetricTile(
                "阈值",
                if (result.threshold.isNaN()) "--" else result.threshold.formatDistance(),
                MaterialTheme.colorScheme.tertiary,
                modifier = Modifier.weight(1f),
              )
            }
          }
        }
        ProcessFlowCard(
          title = "🔬 处理过程",
          rowLabels = listOf("左图", "右图"),
          badgeText = if (result.fallback == "server_yolo") "SERVER YOLO" else "PIPELINE",
          rows =
            listOf(
              listOf(
                ProcessStep("原图", imageA),
                ProcessStep("眼部裁剪", eyeCropA),
                ProcessStep("虹膜区域", irisRegionA),
                ProcessStep("虹膜归一化", normalizedA, wide = true),
              ),
              listOf(
                ProcessStep("原图", imageB),
                ProcessStep("眼部裁剪", eyeCropB),
                ProcessStep("虹膜区域", irisRegionB),
                ProcessStep("虹膜归一化", normalizedB, wide = true),
              ),
            ),
        )
      }
    }
  }
}

@Composable
private fun ComparePreviewProcessCard(imageA: Bitmap?, imageB: Bitmap?, preview: CompareProcessPreview) {
  ProcessFlowCard(
    title = "🔬 已完成处理过程",
    rowLabels = listOf("左图", "右图"),
    badgeText = "CLIENT CROP",
    rows =
      listOf(
        listOf(
          ProcessStep("原图", imageA),
          ProcessStep("眼部裁剪", preview.eyeCropA),
        ),
        listOf(
          ProcessStep("原图", imageB),
          ProcessStep("眼部裁剪", preview.eyeCropB),
        ),
      ),
  )
}

internal fun Double.formatDistance(): String = String.format(Locale.US, "%.4f", this)
