package com.pigeonvision.app

import android.content.Context
import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.ceil
import kotlin.math.floor

data class DetectionResult(
  val left: Float,
  val top: Float,
  val right: Float,
  val bottom: Float,
  val confidence: Float,
) {
  fun expandedCropBounds(
    imageWidth: Int,
    imageHeight: Int,
    paddingRatio: Float = 0.10f,
  ): CropBounds? = expandCropBounds(this, imageWidth, imageHeight, paddingRatio)
}

data class CropBounds(val left: Int, val top: Int, val right: Int, val bottom: Int) {
  val width: Int
    get() = right - left

  val height: Int
    get() = bottom - top
}

data class EyeCrop(val jpegBytes: ByteArray, val detection: DetectionResult)

class YoloDetector(private val context: Context) {
  @Volatile private var loaded = false

  @Synchronized
  fun loadModel(): Boolean {
    if (loaded) return true
    val param = copyAssetToFiles("yolo/model.ncnn.param")
    val bin = copyAssetToFiles("yolo/model.ncnn.bin")
    loaded = nativeLoadModel(param.absolutePath, bin.absolutePath)
    return loaded
  }

  fun detect(
    bitmap: Bitmap,
    confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    nmsThreshold: Float = DEFAULT_NMS_THRESHOLD,
  ): DetectionResult? {
    check(loadModel()) { "YOLO 模型加载失败" }
    if (bitmap.width <= 0 || bitmap.height <= 0 || bitmap.isRecycled) return null

    val input =
      if (bitmap.config == Bitmap.Config.ARGB_8888) {
        bitmap
      } else {
        bitmap.copy(Bitmap.Config.ARGB_8888, false)
      }

    return try {
      nativeDetect(input, confidenceThreshold, nmsThreshold)?.toDetectionResult()
    } finally {
      if (input !== bitmap) input.recycle()
    }
  }

  fun detectAndCropJpeg(
    bitmap: Bitmap,
    confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    nmsThreshold: Float = DEFAULT_NMS_THRESHOLD,
    jpegQuality: Int = JPEG_QUALITY,
  ): EyeCrop? {
    val detection = detect(bitmap, confidenceThreshold, nmsThreshold) ?: return null
    val bounds = detection.expandedCropBounds(bitmap.width, bitmap.height) ?: return null
    val crop = Bitmap.createBitmap(bitmap, bounds.left, bounds.top, bounds.width, bounds.height)
    val bytes =
      ByteArrayOutputStream().use { output ->
        crop.compress(Bitmap.CompressFormat.JPEG, jpegQuality.coerceIn(1, 100), output)
        output.toByteArray()
      }
    crop.recycle()
    return EyeCrop(bytes, detection)
  }

  private fun FloatArray.toDetectionResult(): DetectionResult? {
    if (size < 5) return null
    if (this[2] <= this[0] || this[3] <= this[1]) return null
    return DetectionResult(left = this[0], top = this[1], right = this[2], bottom = this[3], confidence = this[4])
  }

  private fun copyAssetToFiles(assetPath: String): File {
    val outFile = File(context.filesDir, assetPath)
    outFile.parentFile?.mkdirs()
    if (outFile.isFile && outFile.length() > 0L) return outFile
    context.assets.open(assetPath).use { input ->
      outFile.outputStream().use { output -> input.copyTo(output) }
    }
    return outFile
  }

  private external fun nativeLoadModel(paramPath: String, binPath: String): Boolean

  private external fun nativeDetect(bitmap: Bitmap, confidenceThreshold: Float, nmsThreshold: Float): FloatArray?

  companion object {
    const val DEFAULT_CONFIDENCE_THRESHOLD = 0.5f
    const val DEFAULT_NMS_THRESHOLD = 0.45f
    private const val JPEG_QUALITY = 90

    init {
      System.loadLibrary("pigeon_yolo")
    }
  }
}

internal fun expandCropBounds(
  detection: DetectionResult,
  imageWidth: Int,
  imageHeight: Int,
  paddingRatio: Float = 0.10f,
): CropBounds? {
  if (imageWidth <= 0 || imageHeight <= 0) return null
  val boxWidth = detection.right - detection.left
  val boxHeight = detection.bottom - detection.top
  if (boxWidth <= 1f || boxHeight <= 1f) return null

  val padX = boxWidth * paddingRatio
  val padY = boxHeight * paddingRatio
  val left = floor(detection.left - padX).toInt().coerceIn(0, imageWidth - 1)
  val top = floor(detection.top - padY).toInt().coerceIn(0, imageHeight - 1)
  val right = ceil(detection.right + padX).toInt().coerceIn(left + 1, imageWidth)
  val bottom = ceil(detection.bottom + padY).toInt().coerceIn(top + 1, imageHeight)
  return CropBounds(left, top, right, bottom)
}
