import 'dart:ui';
import 'dart:math' as math;

import 'package:flame/cache.dart';
import 'package:flame/components.dart';
import 'package:flame/game.dart';
import 'package:flame/sprite.dart';

class GragasGame extends FlameGame {
  SpriteAnimationComponent? _gragas;

  GragasGame() {
    images = Images(prefix: 'assets/');
  }

  @override
  Color backgroundColor() => const Color(0x00000000);

  @override
  Future<void> onLoad() async {
    await super.onLoad();

    final sheet = SpriteSheet(
      image: await images.load('Gragas32-Sheet.png'),
      srcSize: Vector2.all(48),
    );
    final animation = SpriteAnimation.spriteList(
      List.generate(12, (index) => sheet.getSprite(0, index)),
      stepTime: 0.14,
    );

    _gragas = SpriteAnimationComponent(
      animation: animation,
      anchor: Anchor.center,
    );
    add(_gragas!);
    _layoutGragas();
  }

  @override
  void onGameResize(Vector2 size) {
    super.onGameResize(size);
    _layoutGragas();
  }

  void _layoutGragas() {
    final gragas = _gragas;
    if (gragas == null || size.x <= 0 || size.y <= 0) return;

    final spriteSize = math.min(size.x * 0.8, size.y * 0.86);
    gragas
      ..size = Vector2.all(spriteSize)
      ..position = Vector2(size.x / 2, size.y / 2);
  }
}
