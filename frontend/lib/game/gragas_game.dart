import 'dart:ui';

import 'package:flame/cache.dart';
import 'package:flame/components.dart';
import 'package:flame/game.dart';
import 'package:flame/sprite.dart';

class GragasGame extends FlameGame {
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

    add(
      SpriteAnimationComponent(
        animation: animation,
        size: Vector2.all(240),
        anchor: Anchor.center,
        position: size / 2,
      ),
    );
  }

  @override
  void onGameResize(Vector2 size) {
    super.onGameResize(size);
    for (final component in children.whereType<SpriteAnimationComponent>()) {
      component.position = size / 2;
    }
  }
}
