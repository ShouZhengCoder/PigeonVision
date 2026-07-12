package com.pigeonvision.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class YoloDetectorTest {
  @Test
  fun expandedCropBounds_clampsPaddingToImage() {
    val detection = DetectionResult(left = 0f, top = 10f, right = 100f, bottom = 110f, confidence = 0.9f)

    val bounds = detection.expandedCropBounds(imageWidth = 120, imageHeight = 120)

    assertEquals(CropBounds(left = 0, top = 0, right = 110, bottom = 120), bounds)
  }

  @Test
  fun expandedCropBounds_rejectsInvalidImageSize() {
    val detection = DetectionResult(left = 10f, top = 10f, right = 20f, bottom = 20f, confidence = 0.9f)

    assertNull(detection.expandedCropBounds(imageWidth = 0, imageHeight = 120))
  }
}
