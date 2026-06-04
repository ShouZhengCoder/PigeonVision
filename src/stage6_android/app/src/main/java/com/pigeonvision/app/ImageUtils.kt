package com.pigeonvision.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.math.ceil
import kotlin.math.max

fun decodeBitmapFromUri(context: Context, uri: Uri, maxDimension: Int = 2048): Bitmap {
  return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
      val width = info.size.width
      val height = info.size.height
      val largest = max(width, height)
      if (largest > maxDimension) {
        decoder.setTargetSampleSize(ceil(largest.toDouble() / maxDimension).toInt())
      }
      decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
    }
  } else {
    val options =
      BitmapFactory.Options().apply {
        inJustDecodeBounds = true
      }
    context.contentResolver.openInputStream(uri)?.use { stream ->
      BitmapFactory.decodeStream(stream, null, options)
    }
    val sampleSize = calculateSampleSize(options.outWidth, options.outHeight, maxDimension)
    val decodeOptions =
      BitmapFactory.Options().apply {
        inPreferredConfig = Bitmap.Config.ARGB_8888
        inSampleSize = sampleSize
      }
    context.contentResolver.openInputStream(uri)?.use { stream ->
      BitmapFactory.decodeStream(stream, null, decodeOptions)
    } ?: throw IOException("无法读取图片")
  }
}

fun Bitmap.toJpegBytes(quality: Int = 90): ByteArray =
  ByteArrayOutputStream().use { output ->
    compress(Bitmap.CompressFormat.JPEG, quality.coerceIn(1, 100), output)
    output.toByteArray()
  }

private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
  val largest = max(width, height)
  if (largest <= 0 || largest <= maxDimension) return 1
  return ceil(largest.toDouble() / maxDimension).toInt().coerceAtLeast(1)
}
