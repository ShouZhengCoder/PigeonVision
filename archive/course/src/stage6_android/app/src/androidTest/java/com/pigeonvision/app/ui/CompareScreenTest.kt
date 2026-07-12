package com.pigeonvision.app.ui

import androidx.activity.ComponentActivity
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import com.pigeonvision.app.theme.PigeonVisionTheme
import org.junit.Rule
import org.junit.Test

class CompareScreenTest {
  @get:Rule val composeRule = createAndroidComposeRule<ComponentActivity>()

  @Test
  fun compareScreen_showsRequiredControls() {
    composeRule.setContent {
      PigeonVisionTheme {
        CompareScreen(
          imageA = null,
          imageB = null,
          state = ActionState.Idle,
          onPickA = {},
          onCameraA = {},
          onPickB = {},
          onCameraB = {},
          onCompare = {},
        )
      }
    }

    composeRule.onNodeWithText("图片 A").assertExists()
    composeRule.onNodeWithText("图片 B").assertExists()
    composeRule.onNodeWithText("开始比对").assertExists().assertIsEnabled()
  }
}
