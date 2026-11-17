import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flame/game.dart';
import 'package:frontend/game/gragas_game.dart';

class GragasMapButton extends SpriteAnimationGroupComponent<GragasAnimation>
    with TapCallbacks, HasGameReference<FlameGame> {
  GragasMapButton({
    required super.position,
    required this.onPressed,
  }) : super(
          size: Vector2.all(48),
          anchor: Anchor.center,
          current: GragasAnimation.idle,
        );

  final VoidCallback onPressed;

  @override
  Future<void> onLoad() async {
    animations = await loadGragasAnimations(game.images);
    await super.onLoad();
  }

  @override
  void onTapUp(TapUpEvent event) {
    current = GragasAnimation.throwBarrel;
    animationTicker?.reset();
  }

  @override
  void update(double dt) {
    super.update(dt);
    if (current != GragasAnimation.idle && animationTicker?.done() == true) {
      current = GragasAnimation.idle;
      onPressed();
    }
  }
}
