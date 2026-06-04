package com.pigeonvision.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

internal data class ProcessStep(
  val label: String,
  val bitmap: Bitmap?,
  val wide: Boolean = false,
)

@Composable
internal fun ProcessFlowCard(
  title: String,
  rows: List<List<ProcessStep>>,
  rowLabels: List<String> = emptyList(),
  badgeText: String = "PIPELINE",
  modifier: Modifier = Modifier,
) {
  OutlinedCard(
    modifier = modifier,
    shape = RoundedCornerShape(8.dp),
    colors = CardDefaults.outlinedCardColors(containerColor = MaterialTheme.colorScheme.surface),
  ) {
    Column(
      modifier = Modifier.padding(12.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Column {
          Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
          Text("图像处理链路追踪", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Surface(shape = RoundedCornerShape(999.dp), color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.12f)) {
          Text(
            badgeText,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.tertiary,
            fontWeight = FontWeight.Bold,
          )
        }
      }
      rows.forEachIndexed { index, steps ->
        rowLabels.getOrNull(index)?.takeIf { it.isNotBlank() }?.let { label ->
          Surface(shape = RoundedCornerShape(999.dp), color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)) {
            Text(
              label,
              modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
              style = MaterialTheme.typography.labelMedium,
              color = MaterialTheme.colorScheme.primary,
              fontWeight = FontWeight.Bold,
            )
          }
        }
        ProcessStepRow(steps)
      }
    }
  }
}

@Composable
private fun ProcessStepRow(steps: List<ProcessStep>) {
  Row(
    modifier = Modifier.horizontalScroll(rememberScrollState()),
    horizontalArrangement = Arrangement.spacedBy(8.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    steps.forEachIndexed { index, step ->
      ProcessStepTile(step = step, stepNumber = index + 1)
      if (index < steps.lastIndex) {
        Icon(
          Icons.AutoMirrored.Filled.ArrowForward,
          contentDescription = null,
          tint = MaterialTheme.colorScheme.primary,
        )
      }
    }
  }
}

@Composable
private fun ProcessStepTile(step: ProcessStep, stepNumber: Int) {
  val tileWidth = if (step.wide) 640.dp else 104.dp
  val tileHeight = 80.dp
  val imageModifier =
    Modifier
      .width(tileWidth)
      .height(tileHeight)
      .clip(RoundedCornerShape(8.dp))
      .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(8.dp))
      .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f))

  Column(
    verticalArrangement = Arrangement.spacedBy(6.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    Surface(shape = RoundedCornerShape(999.dp), color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)) {
      Text(
        "STEP ${stepNumber.toString().padStart(2, '0')}",
        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.secondary,
        fontWeight = FontWeight.Bold,
      )
    }
    Box(modifier = imageModifier, contentAlignment = Alignment.Center) {
      val bitmap = step.bitmap
      if (bitmap != null) {
        Image(
          bitmap = bitmap.asImageBitmap(),
          contentDescription = step.label,
          modifier = Modifier.matchParentSize(),
          contentScale = if (step.wide) ContentScale.FillBounds else ContentScale.Crop,
        )
      } else {
        Text(
          "暂无图像",
          color = MaterialTheme.colorScheme.onSurfaceVariant,
          style = MaterialTheme.typography.bodySmall,
        )
      }
    }
    Text(
      step.label,
      style = MaterialTheme.typography.labelMedium,
      color = MaterialTheme.colorScheme.onSurfaceVariant,
      maxLines = 1,
      overflow = TextOverflow.Ellipsis,
      textAlign = TextAlign.Center,
      modifier = Modifier.width(tileWidth),
    )
  }
}
