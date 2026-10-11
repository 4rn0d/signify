import 'dart:ui';
import 'dart:math' as math;

import 'package:flame/cache.dart';
import 'package:flame/components.dart';
import 'package:flame/game.dart';
import 'package:flame/sprite.dart';

enum GragasAnimation { idle, walk, rollBarrel, drinkBarrel, dash, throwBarrel }

extension on GragasAnimation {
  int get row => switch (this) {
    GragasAnimation.idle => 0,
    GragasAnimation.walk => 1,
    GragasAnimation.rollBarrel => 2,
    GragasAnimation.drinkBarrel => 3,
    GragasAnimation.dash => 4,
    GragasAnimation.throwBarrel => 5,
  };

  int get frameCount => switch (this) {
    GragasAnimation.idle => 12,
    GragasAnimation.walk => 8,
    GragasAnimation.rollBarrel => 14,
    GragasAnimation.drinkBarrel => 25,
    GragasAnimation.dash => 14,
    GragasAnimation.throwBarrel => 18,
  };

  bool get loops => switch (this) {
    GragasAnimation.idle || GragasAnimation.walk => true,
    GragasAnimation.rollBarrel ||
    GragasAnimation.dash ||
    GragasAnimation.drinkBarrel ||
    GragasAnimation.throwBarrel => false,
  };
}

class GragasGame extends FlameGame {
  GragasGame({GragasAnimation animation = GragasAnimation.idle})
    : _currentAnimation = animation {
    images = Images(prefix: 'assets/');
  }

  GragasAnimation _currentAnimation;
  SpriteAnimationGroupComponent<GragasAnimation>? _gragas;

  void setAnimation(GragasAnimation animation) {
    _currentAnimation = animation;
    _gragas?.current = animation;
    _gragas?.animationTicker?.reset();
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
    final animations = {
      for (final animation in GragasAnimation.values)
        animation: SpriteAnimation.spriteList(
          List.generate(
            animation.frameCount,
            (index) => sheet.getSprite(animation.row, index),
          ),
          stepTime: 0.14,
          loop: animation.loops,
        ),
    };

    _gragas = SpriteAnimationGroupComponent<GragasAnimation>(
      animations: animations,
      current: _currentAnimation,
      anchor: Anchor.center,
    );
    add(_gragas!);
    _layoutGragas();
  }

  @override
  void update(double dt) {
    super.update(dt);
    final gragas = _gragas;
    if (!_currentAnimation.loops && gragas?.animationTicker?.done() == true) {
      setAnimation(GragasAnimation.idle);
    }
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
