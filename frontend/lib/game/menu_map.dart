import 'dart:async';

import 'package:flame/flame.dart';
import 'package:flame/components.dart';
import 'package:flame_tiled/flame_tiled.dart';
import 'package:frontend/game/gragas_map_button.dart';
import 'package:frontend/game/map_action_hotspot.dart';

String sanitizeMenuTmx(String tmx) {
  return tmx.replaceAllMapped(RegExp(r'<tileset\b[^>]*>'), (match) {
    final tileset = match[0]!;
    if (!RegExp(r'\bcolumns="0"').hasMatch(tileset)) return tileset;
    return tileset.replaceAll(RegExp(r'\s+tilecount="\d+"'), '');
  });
}

class Menu extends World {
  Menu({
    required this.onGragasPressed,
    required this.onQuestsPressed,
    required this.onLeaderboardPressed,
    required this.onSettingsPressed,
    required this.onRoomPositionsLoaded,
  });

  final void Function() onGragasPressed;
  final void Function() onQuestsPressed;
  final void Function() onLeaderboardPressed;
  final void Function() onSettingsPressed;
  final void Function(Map<String, Vector2> positions) onRoomPositionsLoaded;
  late TiledComponent menu;

  @override
  Future<void> onLoad() async {
    final tmx = await Flame.bundle.loadString('assets/tiles/menu.tmx');
    final tileMap = await RenderableTiledMap.fromString(
      sanitizeMenuTmx(tmx),
      Vector2.all(16),
    );
    menu = TiledComponent(tileMap);

    add(menu);
    final positionsLayer =
        menu.tileMap.map.layerByName('Positions') as ObjectGroup;
    final roomPositions = {
      for (final object in positionsLayer.objects)
        if (object.isPoint) object.name: Vector2(object.x, object.y),
    };
    if (!roomPositions.containsKey('entrance_room')) {
      throw StateError('The Positions layer must define entrance_room.');
    }
    onRoomPositionsLoaded(roomPositions);

    final buttonLayer =
        menu.tileMap.map.layerByName('MenuButtons') as ObjectGroup;
    final gragasObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'gragas_button',
    );
    add(
      GragasMapButton(
        position: Vector2(gragasObject.x, gragasObject.y),
        onPressed: onGragasPressed,
      ),
    );

    final questsObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'quests',
    );
    add(
      MapActionHotspot.fromPolygon(
        objectPosition: Vector2(questsObject.x, questsObject.y),
        polygonPoints: [
          for (final point in questsObject.polygon)
            Vector2(point.x.toDouble(), point.y.toDouble()),
        ],
        onPressed: onQuestsPressed,
      ),
    );

    final leaderboardObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'leader_board',
    );
    add(
      MapActionHotspot.fromPolygon(
        objectPosition: Vector2(leaderboardObject.x, leaderboardObject.y),
        polygonPoints: [
          for (final point in leaderboardObject.polygon)
            Vector2(point.x.toDouble(), point.y.toDouble()),
        ],
        onPressed: onLeaderboardPressed,
      ),
    );

    final settingsObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'settings',
    );
    add(
      MapActionHotspot.fromPolygon(
        objectPosition: Vector2(settingsObject.x, settingsObject.y),
        polygonPoints: [
          for (final point in settingsObject.polygon)
            Vector2(point.x.toDouble(), point.y.toDouble()),
        ],
        onPressed: onSettingsPressed,
      ),
    );

    await super.onLoad();
  }
}
