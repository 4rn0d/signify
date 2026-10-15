import 'dart:async';
import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/game.dart';
import 'package:frontend/game/menu_map.dart';

class MenuGame extends FlameGame {

  late final CameraComponent cam;

  final menu = Menu();

  @override
  Color backgroundColor() => const Color(0xFFF6F3EB);

  @override
  FutureOr<void> onLoad() {

    cam = CameraComponent.withFixedResolution(
      world: menu,
      width: 180,
      height: 360,
    );

    cam.viewfinder.anchor = Anchor.topLeft;
    cam.viewfinder.position = Vector2.zero();

    addAll([cam,menu]);
    return super.onLoad();
  }
}
