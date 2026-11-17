import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flame/game.dart';
import 'package:frontend/game/gragas_game.dart';
import 'package:frontend/game/tavern_hover_glow.dart';

class GragasMapButton extends SpriteAnimationGroupComponent<GragasAnimation>
    with TapCallbacks, HoverCallbacks, HasGameReference<FlameGame> {
  GragasMapButton({
    required super.position,
    required this.onPressed,
  }) : super(
          size: Vector2.all(48),
          anchor: Anchor.center,
          current: GragasAnimation.idle,
        );

  final VoidCallback onPressed;
  bool _hovered = false;

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
  void onHoverEnter() => _hovered = true;

  @override
  void onHoverExit() => _hovered = false;

  @override
  void render(Canvas canvas) {
    if (_hovered) {
      final path = Path()
        ..addRRect(
          RRect.fromRectAndRadius(
            size.toRect(),
            const Radius.circular(8),
          ),
        );
      renderTavernHoverGlow(canvas, path);
    }
    super.render(canvas);
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
