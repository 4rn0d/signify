import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flame/experimental.dart';
import 'package:flame/game.dart';
import 'package:frontend/game/menu_map.dart';

class MenuGame extends FlameGame with DragCallbacks {
  MenuGame({required VoidCallback onGragasPressed})
      : menu = Menu(onGragasPressed: onGragasPressed);

  static const _mapWidth = 60 * 16.0;
  static const _mapHeight = 60 * 16.0;
  static const _initialViewHeight = 360.0;

  late final CameraComponent cam;
  Vector2? _canvasSize;
  bool _cameraInitialized = false;
  bool _cameraWasDragged = false;

  final Menu menu;

  @override
  Color backgroundColor() => const Color(0xFFF6F3EB);

  @override
  FutureOr<void> onLoad() {
    cam = CameraComponent(world: menu);

    cam.viewfinder.anchor = Anchor.bottomCenter;
    final canvasSize = _canvasSize ?? size;
    cam.viewfinder.zoom = _zoomFor(canvasSize);
    if (canvasSize.x > 0 && canvasSize.y > 0) {
      cam.viewfinder.position = _initialPosition(canvasSize);
    }
    cam.setBounds(
      Rectangle.fromLTWH(0, 0, _mapWidth, _mapHeight),
      considerViewport: true,
    );

    addAll([cam, menu]);
    _cameraInitialized = true;
    return super.onLoad();
  }

  @override
  void onGameResize(Vector2 size) {
    super.onGameResize(size);
    _canvasSize = size.clone();
    if (_cameraInitialized) {
      cam.viewfinder.zoom = _zoomFor(size);
      if (!_cameraWasDragged && size.x > 0 && size.y > 0) {
        cam.viewfinder.position = _initialPosition(size);
      }
    }
  }

  @override
  void onDragUpdate(DragUpdateEvent event) {
    _cameraWasDragged = true;
    final start = cam.viewport.globalToLocal(event.canvasStartPosition);
    final end = cam.viewport.globalToLocal(event.canvasEndPosition);
    cam.viewfinder.position -= (end - start) / cam.viewfinder.zoom;
  }

  double _zoomFor(Vector2 canvasSize) {
    if (canvasSize.x <= 0 || canvasSize.y <= 0) {
      return 1;
    }
    return math.max(
      canvasSize.y / _initialViewHeight,
      canvasSize.x / _mapWidth,
    );
  }

  Vector2 _initialPosition(Vector2 canvasSize) {
    final zoom = _zoomFor(canvasSize);
    return Vector2(
      canvasSize.x / zoom / 2,
      canvasSize.y / zoom / 2,
    );
  }
}
