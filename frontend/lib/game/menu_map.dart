import 'dart:async';
import 'dart:ui';

import 'package:flame/flame.dart';
import 'package:flame/components.dart';
import 'package:flame_tiled/flame_tiled.dart';
import 'package:frontend/game/gragas_map_button.dart';
import 'package:frontend/game/map_action_hotspot.dart';

String sanitizeMenuTmx(String tmx) {
  // Flame.Images already prefixes paths with assets/images/.
  final normalizedImagePaths = tmx.replaceAll(
    'source="../images/',
    'source="',
  );
  return normalizedImagePaths.replaceAllMapped(
    RegExp(r'<tileset\b[^>]*>'),
    (match) {
      final tileset = match[0]!;
      if (!RegExp(r'\bcolumns="0"').hasMatch(tileset)) return tileset;
      return tileset.replaceAll(RegExp(r'\s+tilecount="\d+"'), '');
    },
  );
}

class Menu extends World {
  Menu({
    required this.onGragasPressed,
    required this.onQuestsPressed,
    required this.onLeaderboardPressed,
    required this.onProfilePressed,
    required this.onToKitchenPressed,
    required this.onRoomPositionsLoaded,
  });

  final void Function() onGragasPressed;
  final void Function() onQuestsPressed;
  final void Function() onLeaderboardPressed;
  final void Function() onProfilePressed;
  final void Function() onToKitchenPressed;
  final void Function(Map<String, Vector2> positions) onRoomPositionsLoaded;

  @override
  Future<void> onLoad() async {
    final tmx = await Flame.bundle.loadString('assets/tiles/menu.tmx');
    final tileMap = await RenderableTiledMap.fromString(
      sanitizeMenuTmx(tmx),
      Vector2.all(16),
      // flame_tiled's default layer paint is anti-aliased, which softens every
      // tile quad's edges and lets the background show through as thin seams.
      layerPaintFactory: (opacity) => Paint()
        ..color = Color.fromRGBO(255, 255, 255, opacity)
        ..isAntiAlias = false
        ..filterQuality = FilterQuality.none,
    );
    add(await _bakeMap(tileMap));
    final positionsLayer =
        tileMap.map.layerByName('Positions') as ObjectGroup;
    final roomPositions = {
      for (final object in positionsLayer.objects)
        if (object.isPoint) object.name: Vector2(object.x, object.y),
    };
    if (!roomPositions.containsKey('entrance_room')) {
      throw StateError('The Positions layer must define entrance_room.');
    }
    onRoomPositionsLoaded(roomPositions);

    final buttonLayer =
        tileMap.map.layerByName('MenuButtons') as ObjectGroup;
    final gragasObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'gragas_button',
    );
    add(
      GragasMapButton(
        position: Vector2(gragasObject.x, gragasObject.y)..round(),
        label: 'Practice',
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
        label: 'Quests',
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
        label: 'Leaderboard',
        onPressed: onLeaderboardPressed,
      ),
    );

    final profileObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'profile',
    );
    add(
      MapActionHotspot.fromPolygon(
        objectPosition: Vector2(profileObject.x, profileObject.y),
        polygonPoints: [
          for (final point in profileObject.polygon)
            Vector2(point.x.toDouble(), point.y.toDouble()),
        ],
        label: 'Profile',
        onPressed: onProfilePressed,
      ),
    );

    final toKitchenObject = buttonLayer.objects.singleWhere(
      (object) => object.name == 'to_kitchen',
    );
    add(
      MapActionHotspot.fromPolygon(
        objectPosition: Vector2(toKitchenObject.x, toKitchenObject.y),
        polygonPoints: [
          Vector2(0, 0),
          Vector2(toKitchenObject.width, 0),
          Vector2(toKitchenObject.width, toKitchenObject.height),
          Vector2(0, toKitchenObject.height),
        ],
        label: 'Kitchen',
        onPressed: onToKitchenPressed,
      ),
    );

    await super.onLoad();
  }

  /// Renders every tile layer once, at 1 texel per world unit, into a single
  /// image. Drawing hundreds of separate tile quads under a scaled camera can
  /// leave hairline gaps between neighbouring tiles; one image cannot.
  /// The map has no animated tiles, so a static snapshot is equivalent.
  Future<SpriteComponent> _bakeMap(RenderableTiledMap tileMap) async {
    final map = tileMap.map;
    final width = map.width * map.tileWidth;
    final height = map.height * map.tileHeight;

    final recorder = PictureRecorder();
    tileMap.render(Canvas(recorder));
    final picture = recorder.endRecording();
    final image = await picture.toImage(width, height);
    picture.dispose();

    return SpriteComponent(
      sprite: Sprite(image),
      size: Vector2(width.toDouble(), height.toDouble()),
      paint: Paint()
        ..isAntiAlias = false
        ..filterQuality = FilterQuality.none,
    );
  }
}
