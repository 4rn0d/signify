import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flutter/painting.dart';
import 'package:frontend/game/tavern_hover_glow.dart';

class MapActionHotspot extends PositionComponent
    with TapCallbacks, HoverCallbacks {
  factory MapActionHotspot.fromPolygon({
    required Vector2 objectPosition,
    required List<Vector2> polygonPoints,
    required String label,
    required void Function() onPressed,
  }) {
    if (polygonPoints.length < 3) {
      throw ArgumentError.value(
        polygonPoints,
        'polygonPoints',
        'A hotspot polygon must have at least three points.',
      );
    }

    var minX = polygonPoints.first.x;
    var minY = polygonPoints.first.y;
    var maxX = minX;
    var maxY = minY;
    for (final point in polygonPoints.skip(1)) {
      minX = point.x < minX ? point.x : minX;
      minY = point.y < minY ? point.y : minY;
      maxX = point.x > maxX ? point.x : maxX;
      maxY = point.y > maxY ? point.y : maxY;
    }

    final localPolygon = [
      for (final point in polygonPoints)
        Vector2(point.x - minX, point.y - minY),
    ];
    return MapActionHotspot._(
      position: objectPosition + Vector2(minX, minY),
      size: Vector2(maxX - minX, maxY - minY),
      polygon: localPolygon,
      label: label,
      onPressed: onPressed,
    );
  }

  MapActionHotspot._({
    required super.position,
    required super.size,
    required List<Vector2> polygon,
    required this.label,
    required this.onPressed,
  })  : _polygonPath = _createPolygonPath(polygon),
        super(anchor: Anchor.topLeft);

  final void Function() onPressed;
  final String label;
  final Path _polygonPath;
  bool _hovered = false;

  @override
  void onTapUp(TapUpEvent event) => onPressed();

  @override
  void onHoverEnter() => _hovered = true;

  @override
  void onHoverExit() => _hovered = false;

  @override
  bool containsLocalPoint(Vector2 point) =>
      _polygonPath.contains(Offset(point.x, point.y));

  @override
  void render(Canvas canvas) {
    if (!_hovered) return;

    renderTavernHoverGlow(canvas, _polygonPath, label: label);
  }

  static Path _createPolygonPath(List<Vector2> points) {
    final path = Path()..moveTo(points.first.x, points.first.y);
    for (final point in points.skip(1)) {
      path.lineTo(point.x, point.y);
    }
    return path..close();
  }
}
