package com.pigeonvision.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import java.io.IOException
import java.util.concurrent.TimeUnit
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject

data class HealthResponse(val status: String, val gallerySize: Int)

data class CompareResponse(
  val distance: Double,
  val sameFamily: Boolean,
  val threshold: Double,
  val eyeCropA: String,
  val irisRegionA: String,
  val normalizedA: String,
  val eyeCropB: String,
  val irisRegionB: String,
  val normalizedB: String,
  val fallback: String,
)

data class SearchResponse(
  val results: List<SearchResult>,
  val eyeCrop: String,
  val irisRegion: String,
  val normalized: String,
  val fallback: String,
)

data class SearchResult(
  val rank: Int,
  val imgId: String,
  val bloodId: String,
  val bloodName: String,
  val distance: Double,
  val pgId: String,
  val imageUrl: String,
)

class ApiClient(
  private val client: OkHttpClient =
    OkHttpClient.Builder()
      .connectTimeout(15, TimeUnit.SECONDS)
      .readTimeout(60, TimeUnit.SECONDS)
      .writeTimeout(60, TimeUnit.SECONDS)
      .callTimeout(90, TimeUnit.SECONDS)
      .build(),
) {
  private val jpegMediaType = "image/jpeg".toMediaType()

  suspend fun health(baseUrl: String): HealthResponse =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val request = Request.Builder().url("${normalizeBaseUrl(baseUrl)}/health").get().build()
      client.newCall(request).execute().use { response ->
        parseHealth(response.requireBodyText())
      }
    }

  suspend fun compare(
    baseUrl: String,
    imageA: ByteArray,
    imageB: ByteArray,
    eyeCrop: Boolean = true,
    originalImageA: ByteArray? = null,
    originalImageB: ByteArray? = null,
  ): CompareResponse =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val builder =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart("image_a", "image_a.jpg", imageA.toRequestBody(jpegMediaType))
          .addFormDataPart("image_b", "image_b.jpg", imageB.toRequestBody(jpegMediaType))
          .addFormDataPart("eye_crop", if (eyeCrop) "1" else "0")
      if (originalImageA != null) {
        builder.addFormDataPart("original_image_a", "original_image_a.jpg", originalImageA.toRequestBody(jpegMediaType))
      }
      if (originalImageB != null) {
        builder.addFormDataPart("original_image_b", "original_image_b.jpg", originalImageB.toRequestBody(jpegMediaType))
      }
      val body = builder.build()
      val request = Request.Builder().url("${normalizeBaseUrl(baseUrl)}/compare").post(body).build()
      client.newCall(request).execute().use { response ->
        parseCompare(response.requireBodyText())
      }
    }

  suspend fun search(
    baseUrl: String,
    image: ByteArray,
    topK: Int = DEFAULT_SEARCH_TOP_K,
    eyeCrop: Boolean = true,
    originalImage: ByteArray? = null,
  ): SearchResponse =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val builder =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart("image", "image.jpg", image.toRequestBody(jpegMediaType))
          .addFormDataPart("top_k", topK.toString())
          .addFormDataPart("eye_crop", if (eyeCrop) "1" else "0")
      if (originalImage != null) {
        builder.addFormDataPart("original_image", "original_image.jpg", originalImage.toRequestBody(jpegMediaType))
      }
      val body = builder.build()
      val request = Request.Builder().url("${normalizeBaseUrl(baseUrl)}/search").post(body).build()
      client.newCall(request).execute().use { response ->
        parseSearch(response.requireBodyText())
      }
    }

  suspend fun downloadBytes(url: String): ByteArray =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val request = Request.Builder().url(url).get().build()
      client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) {
          throw IOException("HTTP ${response.code}")
        }
        response.body?.bytes() ?: throw IOException("服务器返回为空")
      }
    }

  private fun Response.requireBodyText(): String {
    val text = body?.string().orEmpty()
    if (!isSuccessful) {
      throw IOException("HTTP $code: ${extractErrorMessage(text).take(220)}")
    }
    if (text.isBlank()) {
      throw IOException("服务器返回为空")
    }
    return text
  }

  companion object {
    const val DEFAULT_SEARCH_TOP_K = 20
    const val MAX_SEARCH_TOP_K = 100

    fun extractErrorMessage(text: String): String {
      if (text.isBlank()) return "服务器返回错误"
      return try {
        val root = JSONObject(text)
        root.optString("error", text).ifBlank { text }
      } catch (_: Throwable) {
        text
      }
    }

    fun normalizeBaseUrl(value: String): String {
      val trimmed = value.trim().trimEnd('/')
      require(trimmed.isNotEmpty()) { "请先输入服务器地址" }
      return if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
        trimmed
      } else {
        "http://$trimmed"
      }
    }

    fun resolveUrl(baseUrl: String, url: String): String {
      val value = url.trim()
      if (value.startsWith("http://") || value.startsWith("https://")) return value
      val base = normalizeBaseUrl(baseUrl)
      return if (value.startsWith("/")) {
        "$base$value"
      } else {
        "$base/$value"
      }
    }

    fun base64ToBitmap(s: String): Bitmap? {
      val value = s.trim()
      if (value.isBlank()) return null
      val payload = value.substringAfter("base64,", value)
      return try {
        val bytes = Base64.decode(payload, Base64.DEFAULT)
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
      } catch (_: Throwable) {
        null
      }
    }

    fun parseHealth(json: String): HealthResponse {
      val root = JSONObject(json)
      return HealthResponse(
        status = root.optString("status", "unknown"),
        gallerySize = root.optInt("gallery_size", 0),
      )
    }

    fun parseCompare(json: String): CompareResponse {
      val root = JSONObject(json)
      return CompareResponse(
        distance = root.getDouble("distance"),
        sameFamily = root.getBoolean("same_family"),
        threshold = root.optDouble("threshold", Double.NaN),
        eyeCropA = root.optString("eye_crop_a", ""),
        irisRegionA = root.optString("iris_region_a", ""),
        normalizedA = root.optString("normalized_a", ""),
        eyeCropB = root.optString("eye_crop_b", ""),
        irisRegionB = root.optString("iris_region_b", ""),
        normalizedB = root.optString("normalized_b", ""),
        fallback = root.optString("fallback", ""),
      )
    }

    fun parseSearch(json: String): SearchResponse {
      val root = JSONObject(json)
      val array = root.getJSONArray("results")
      val results =
        buildList {
          for (index in 0 until array.length()) {
            val item = array.getJSONObject(index)
            add(
              SearchResult(
                rank = item.optInt("rank", index + 1),
                imgId = item.optString("img_id", ""),
                bloodId = item.optString("blood_id", ""),
                bloodName = item.optString("blood_name", "未知品系"),
                distance = item.optDouble("distance", Double.NaN),
                pgId = item.optString("pg_id", "-"),
                imageUrl = item.optString("image_url", ""),
              )
            )
          }
        }
      return SearchResponse(
        results = results,
        eyeCrop = root.optString("eye_crop", ""),
        irisRegion = root.optString("iris_region", ""),
        normalized = root.optString("normalized", ""),
        fallback = root.optString("fallback", ""),
      )
    }
  }
}
