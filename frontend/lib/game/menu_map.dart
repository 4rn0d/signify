import 'dart:async';

import 'package:flame/components.dart';
import 'package:flame_tiled/flame_tiled.dart';

class Menu extends World {

  late TiledComponent menu;

  @override
  FutureOr<void> onLoad() async {

    menu = await TiledComponent.load(
      'menu.tmx',
      Vector2.all(16)
    );

    add(menu);

    return super.onLoad();
  }
}