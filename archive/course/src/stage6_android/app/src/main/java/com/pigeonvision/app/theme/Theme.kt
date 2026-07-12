package com.pigeonvision.app.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors =
  lightColorScheme(
    primary = SeedGreen,
    secondary = RingAmber,
    tertiary = IrisBlue,
    background = PigeonSurface,
    surface = Color.White,
  )

private val DarkColors =
  darkColorScheme(
    primary = Color(0xFF77D7B0),
    secondary = Color(0xFFE3B261),
    tertiary = Color(0xFF8CB9E8),
    background = PigeonSurfaceDark,
    surface = Color(0xFF171C20),
  )

@Composable
fun PigeonVisionTheme(
  darkTheme: Boolean = isSystemInDarkTheme(),
  content: @Composable () -> Unit,
) {
  MaterialTheme(colorScheme = if (darkTheme) DarkColors else LightColors, typography = Typography, content = content)
}
