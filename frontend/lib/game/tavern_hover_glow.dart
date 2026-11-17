import 'package:flutter/painting.dart';
import 'package:frontend/theme/app_theme.dart';

void renderTavernHoverGlow(Canvas canvas, Path path) {
  canvas.drawPath(
    path,
    Paint()
      ..color = TavernPalette.parchment.withValues(alpha: 0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 12
      ..strokeJoin = StrokeJoin.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10),
  );
  canvas.drawPath(
    path,
    Paint()
      ..color = TavernPalette.parchment.withValues(alpha: 0.38)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeJoin = StrokeJoin.round,
  );
}
