import 'package:flutter/painting.dart';
import 'package:frontend/theme/app_theme.dart';

void renderTavernHoverGlow(
  Canvas canvas,
  Path path, {
  required String label,
}) {
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

  final textPainter = TextPainter(
    text: TextSpan(
      text: label,
      style: const TextStyle(
        color: TavernPalette.ink,
        fontFamily: 'serif',
        fontSize: 6,
        fontWeight: FontWeight.bold,
      ),
    ),
    textDirection: TextDirection.ltr,
  )..layout();
  final bounds = path.getBounds();
  final labelRect = Rect.fromLTWH(
    bounds.center.dx - (textPainter.width + 15) / 2,
    bounds.top - textPainter.height - 13,
    textPainter.width + 15,
    textPainter.height + 5,
  );
  final labelShape = RRect.fromRectAndRadius(
    labelRect,
    const Radius.circular(3),
  );
  final labelPath = Path()..addRRect(labelShape);
  canvas.drawShadow(
    labelPath,
    TavernPalette.wood.withValues(alpha: 0.45),
    4,
    true,
  );
  canvas.drawRRect(
    labelShape,
    Paint()..color = TavernPalette.parchment,
  );
  canvas.drawRRect(
    labelShape,
    Paint()
      ..color = TavernPalette.brass
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5,
  );
  textPainter.paint(
    canvas,
    Offset(labelRect.left + 7, labelRect.top + 2),
  );
}
