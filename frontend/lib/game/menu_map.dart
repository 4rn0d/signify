import 'dart:async';

import 'package:flame/components.dart';
import 'package:flame_tiled/flame_tiled.dart';
import 'package:frontend/game/gragas_map_button.dart';

class Menu extends World {
  Menu({required this.onGragasPressed});

  final void Function() onGragasPressed;
  late TiledComponent menu;

  @override
  Future<void> onLoad() async {
    menu = await TiledComponent.load(
      'menu.tmx',
      Vector2.all(16),
    );

    add(menu);
    final buttonLayer =
        menu.tileMap.map.layerByName('MenuButtons') as ObjectGroup;
    final gragasObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'gragas_button',
    );
    add(
      GragasMapButton(
        position: Vector2(gragasObject.x, gragasObject.y),
        onPressed: onGragasPressed
      ),
    );

    await super.onLoad();
  }
}
