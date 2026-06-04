package com.pigeonvision.app

import org.junit.Assert.assertEquals
import org.junit.Test

class ApiClientTest {
  @Test
  fun parseCompare_readsProcessImageFields() {
    val response =
      ApiClient.parseCompare(
        """
        {
          "distance": 0.83,
          "same_family": true,
          "threshold": 2.348,
          "eye_crop_a": "crop-a",
          "iris_region_a": "iris-a",
          "normalized_a": "norm-a",
          "eye_crop_b": "crop-b",
          "iris_region_b": "iris-b",
          "normalized_b": "norm-b",
          "fallback": "server_yolo"
        }
        """.trimIndent()
      )

    assertEquals(0.83, response.distance, 0.0)
    assertEquals(true, response.sameFamily)
    assertEquals(2.348, response.threshold, 0.0)
    assertEquals("crop-a", response.eyeCropA)
    assertEquals("iris-a", response.irisRegionA)
    assertEquals("norm-a", response.normalizedA)
    assertEquals("crop-b", response.eyeCropB)
    assertEquals("iris-b", response.irisRegionB)
    assertEquals("norm-b", response.normalizedB)
    assertEquals("server_yolo", response.fallback)
  }

  @Test
  fun parseSearch_readsImageFields() {
    val response =
      ApiClient.parseSearch(
        """
        {
          "eye_crop": "query-crop",
          "iris_region": "query-iris",
          "normalized": "query-norm",
          "fallback": "server_yolo",
          "results": [
            {
              "rank": 1,
              "img_id": "100009",
              "blood_id": "B01",
              "blood_name": "小迪克孙",
              "distance": 0.21,
              "pg_id": "NL15-1273729",
              "image_url": "/image/100009"
            }
          ]
        }
        """.trimIndent()
      )

    assertEquals("query-crop", response.eyeCrop)
    assertEquals("query-iris", response.irisRegion)
    assertEquals("query-norm", response.normalized)
    assertEquals("server_yolo", response.fallback)
    val result = response.results.single()
    assertEquals(1, result.rank)
    assertEquals("100009", result.imgId)
    assertEquals("B01", result.bloodId)
    assertEquals("小迪克孙", result.bloodName)
    assertEquals("NL15-1273729", result.pgId)
    assertEquals("/image/100009", result.imageUrl)
  }

  @Test
  fun resolveUrl_acceptsRelativeAndAbsoluteUrls() {
    assertEquals("http://192.168.1.2:5000/image/100009", ApiClient.resolveUrl("192.168.1.2:5000/", "/image/100009"))
    assertEquals("https://example.com/image/1", ApiClient.resolveUrl("http://server", "https://example.com/image/1"))
  }

  @Test
  fun extractErrorMessage_readsServerErrorJson() {
    assertEquals(
      "处理失败：虹膜分割失败：ellipse centers too far apart",
      ApiClient.extractErrorMessage("""{"error":"处理失败：虹膜分割失败：ellipse centers too far apart"}"""),
    )
  }
}
