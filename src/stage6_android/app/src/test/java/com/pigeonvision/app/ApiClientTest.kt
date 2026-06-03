package com.pigeonvision.app

import org.junit.Assert.assertEquals
import org.junit.Test

class ApiClientTest {
  @Test
  fun parseSearch_readsImageFields() {
    val response =
      ApiClient.parseSearch(
        """
        {
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
}
