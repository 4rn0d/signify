import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flame/game.dart';
import 'package:frontend/game/menu_map.dart';

class MenuGame extends FlameGame with DragCallbacks {
  MenuGame({required VoidCallback onGragasPressed}) {
    menu = Menu(
      onGragasPressed: onGragasPressed,
      onRoomPositionsLoaded: _setRoomPositions,
    );
  }

  static const _roomViewWidth = 820.0;
  static const _tileSize = 16.0;
  static const _swipeThreshold = 48.0;
  static const _roomTransitionDuration = 0.35;

  late final CameraComponent cam;
  bool _cameraInitialized = false;
  late final Menu menu;
  Map<String, Vector2> _roomPositions = {};
  String _currentRoom = 'entrance_room';
  Vector2? _dragStartPosition;
  bool _swipeHandled = false;
  Vector2? _transitionStart;
  Vector2? _transitionEnd;
  double _transitionElapsed = 0;

  @override
  Color backgroundColor() => const Color(0xFFF6F3EB);

  @override
  FutureOr<void> onLoad() {
    cam = CameraComponent(world: menu);

    cam.viewfinder.anchor = Anchor.center;
    final canvasSize = size;
    cam.viewfinder.zoom = _zoomFor(canvasSize);

    addAll([cam, menu]);
    _cameraInitialized = true;
    return super.onLoad();
  }

  @override
  void onGameResize(Vector2 size) {
    super.onGameResize(size);
    if (_cameraInitialized) {
      cam.viewfinder.zoom = _zoomFor(size);
      cam.viewfinder.position = _pixelAlignedPosition(
        cam.viewfinder.position,
        size,
      );
    }
  }

  @override
  void update(double dt) {
    super.update(dt);
    final start = _transitionStart;
    final end = _transitionEnd;
    if (start == null || end == null) return;

    _transitionElapsed += dt;
    final progress = (_transitionElapsed / _roomTransitionDuration)
        .clamp(0.0, 1.0)
        .toDouble();
    final easedProgress = progress * (2 - progress);
    cam.viewfinder.position.setFrom(
      _pixelAlignedPosition(start + (end - start) * easedProgress, size),
    );

    if (progress >= 1) {
      _transitionStart = null;
      _transitionEnd = null;
    }
  }

  @override
  void onDragStart(DragStartEvent event) {
    super.onDragStart(event);
    _dragStartPosition = event.canvasPosition.clone();
    _swipeHandled = false;
  }

  @override
  void onDragUpdate(DragUpdateEvent event) {
    if (_swipeHandled || _transitionStart != null) return;

    final start = _dragStartPosition;
    if (start == null) return;
    final displacement = event.canvasEndPosition - start;
    if (displacement.x.abs() < _swipeThreshold ||
        displacement.x.abs() < displacement.y.abs() * 1.3) {
      return;
    }

    _swipeHandled = true;
    _navigateHorizontally(displacement.x);
  }

  @override
  void onDragEnd(DragEndEvent event) {
    super.onDragEnd(event);
    _dragStartPosition = null;
  }

  void _setRoomPositions(Map<String, Vector2> positions) {
    _roomPositions = positions;
    cam.viewfinder.position = _pixelAlignedPosition(
      positions[_currentRoom]!,
      size,
    );
  }

  void _navigateHorizontally(double swipeDeltaX) {
    final currentPosition = _roomPositions[_currentRoom];
    if (currentPosition == null) return;

    // Swiping right reveals the room to the left, like a horizontal pager.
    final targetXDirection = swipeDeltaX > 0 ? -1 : 1;
    final candidates = _roomPositions.entries.where((entry) {
      final dx = entry.value.x - currentPosition.x;
      return entry.key != _currentRoom && dx * targetXDirection > 0;
    }).toList()
      ..sort((a, b) {
        final distanceA = (a.value.x - currentPosition.x).abs();
        final distanceB = (b.value.x - currentPosition.x).abs();
        return distanceA.compareTo(distanceB);
      });

    if (candidates.isEmpty) return;

    _currentRoom = candidates.first.key;
    _transitionStart = cam.viewfinder.position.clone();
    _transitionEnd = _pixelAlignedPosition(candidates.first.value, size);
    _transitionElapsed = 0;
  }

  double _zoomFor(Vector2 canvasSize) {
    if (canvasSize.x <= 0 || canvasSize.y <= 0) {
      return 1;
    }
    final pixelRatio = _devicePixelRatio;
    final desiredZoom = canvasSize.x / _roomViewWidth;
    final tilePixels = math.max(
      1,
      (desiredZoom * _tileSize * pixelRatio).round(),
    );
    return tilePixels / (_tileSize * pixelRatio);
  }

  Vector2 _pixelAlignedPosition(Vector2 position, Vector2 canvasSize) {
    final pixelsPerWorldUnit = cam.viewfinder.zoom * _devicePixelRatio;
    final centerX = canvasSize.x * _devicePixelRatio / 2;
    final centerY = canvasSize.y * _devicePixelRatio / 2;

    return Vector2(
      (centerX - (centerX - position.x * pixelsPerWorldUnit).round()) /
          pixelsPerWorldUnit,
      (centerY - (centerY - position.y * pixelsPerWorldUnit).round()) /
          pixelsPerWorldUnit,
    );
  }

  double get _devicePixelRatio {
    final views = PlatformDispatcher.instance.views;
    return views.isEmpty ? 1 : views.first.devicePixelRatio;
  }
}
