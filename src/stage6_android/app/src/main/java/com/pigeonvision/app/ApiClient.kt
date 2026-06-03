package com.pigeonvision.app

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
)

data class SearchResponse(val results: List<SearchResult>)

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

  suspend fun compare(baseUrl: String, imageA: ByteArray, imageB: ByteArray): CompareResponse =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val body =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart("image_a", "image_a.jpg", imageA.toRequestBody(jpegMediaType))
          .addFormDataPart("image_b", "image_b.jpg", imageB.toRequestBody(jpegMediaType))
          .addFormDataPart("eye_crop", "1")
          .build()
      val request = Request.Builder().url("${normalizeBaseUrl(baseUrl)}/compare").post(body).build()
      client.newCall(request).execute().use { response ->
        parseCompare(response.requireBodyText())
      }
    }

  suspend fun search(baseUrl: String, image: ByteArray, topK: Int = DEFAULT_SEARCH_TOP_K): SearchResponse =
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
      val body =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart("image", "image.jpg", image.toRequestBody(jpegMediaType))
          .addFormDataPart("top_k", topK.toString())
          .addFormDataPart("eye_crop", "1")
          .build()
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
      throw IOException("HTTP $code: ${text.take(180)}")
    }
    if (text.isBlank()) {
      throw IOException("服务器返回为空")
    }
    return text
  }

  companion object {
    const val DEFAULT_SEARCH_TOP_K = 20
    const val MAX_SEARCH_TOP_K = 100

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
      return SearchResponse(results)
    }
  }
}
