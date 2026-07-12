package com.pigeonvision.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SearchImageLoader(
  private val apiClient: ApiClient,
  maxEntries: Int = 80,
) {
  private val cache = LruCache<String, Bitmap>(maxEntries)

  suspend fun load(baseUrl: String, imageUrl: String): Bitmap {
    val resolvedUrl = ApiClient.resolveUrl(baseUrl, imageUrl)
    cache.get(resolvedUrl)?.let { return it }

    val bytes = apiClient.downloadBytes(resolvedUrl)
    val bitmap =
      withContext(Dispatchers.Default) {
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("图片解码失败")
      }
    cache.put(resolvedUrl, bitmap)
    return bitmap
  }
}
