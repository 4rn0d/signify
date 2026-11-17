import 'package:flutter/material.dart';

abstract final class TavernPalette {
  static const wood = Color(0xFF5C3825);
  static const woodLight = Color(0xFF805236);
  static const brass = Color(0xFFAA762E);
  static const parchment = Color(0xFFF2E5C9);
  static const parchmentLight = Color(0xFFFAF2E2);
  static const canvas = Color(0xFFE6D4B0);
  static const ink = Color(0xFF2E2118);
  static const sage = Color(0xFF596344);
}

abstract final class AppTheme {
  static ThemeData get light {
    final colors = ColorScheme.fromSeed(
      seedColor: TavernPalette.wood,
      surface: TavernPalette.parchment,
      brightness: Brightness.light,
    ).copyWith(
      primary: TavernPalette.wood,
      onPrimary: TavernPalette.parchmentLight,
      secondary: TavernPalette.brass,
      onSecondary: TavernPalette.ink,
      tertiary: TavernPalette.sage,
      onTertiary: TavernPalette.parchmentLight,
      surface: TavernPalette.parchment,
      onSurface: TavernPalette.ink,
      outline: TavernPalette.woodLight,
      error: const Color(0xFF8C3028),
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colors,
      fontFamily: 'serif',
      scaffoldBackgroundColor: TavernPalette.canvas,
      appBarTheme: const AppBarTheme(
        backgroundColor: TavernPalette.canvas,
        foregroundColor: TavernPalette.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
      cardTheme: CardThemeData(
        color: TavernPalette.parchment,
        surfaceTintColor: Colors.transparent,
        elevation: 2,
        shadowColor: TavernPalette.wood.withValues(alpha: 0.22),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(
            color: TavernPalette.wood.withValues(alpha: 0.3),
          ),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: TavernPalette.wood,
          foregroundColor: TavernPalette.parchmentLight,
          minimumSize: const Size(0, 48),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
            side: const BorderSide(color: TavernPalette.woodLight),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: TavernPalette.parchmentLight,
        labelStyle: const TextStyle(color: TavernPalette.wood),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: TavernPalette.woodLight),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(
            color: TavernPalette.wood.withValues(alpha: 0.45),
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: TavernPalette.brass, width: 2),
        ),
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: TavernPalette.brass,
      ),
      snackBarTheme: const SnackBarThemeData(
        backgroundColor: TavernPalette.wood,
        contentTextStyle: TextStyle(color: TavernPalette.parchmentLight),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
