import 'package:flame/game.dart';
import 'package:flutter/widgets.dart';

import '../game/gragas_game.dart';

export '../game/gragas_game.dart' show GragasAnimation;

class GragasAvatar extends StatefulWidget {
  const GragasAvatar({
    super.key,
    this.animation = GragasAnimation.idle,
    this.animationTrigger = 0,
  });

  final GragasAnimation animation;
  final int animationTrigger;

  @override
  State<GragasAvatar> createState() => _GragasAvatarState();
}

class _GragasAvatarState extends State<GragasAvatar> {
  late final GragasGame _game;

  @override
  void initState() {
    super.initState();
    _game = GragasGame(animation: widget.animation);
  }

  @override
  void didUpdateWidget(covariant GragasAvatar oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.animation != widget.animation ||
        oldWidget.animationTrigger != widget.animationTrigger) {
      _game.setAnimation(widget.animation);
    }
  }

  @override
  Widget build(BuildContext context) {
    return GameWidget(
      game: _game,
      backgroundBuilder: (context) => const SizedBox.expand(),
    );
  }
}
