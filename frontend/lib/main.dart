import 'package:flame/game.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:frontend/game/menu_game.dart';
import 'package:frontend/pages/training_page.dart';
import 'package:frontend/theme/app_theme.dart';

import 'auth_gate.dart';
import 'firebase_options.dart';
import 'pages/home_page.dart';
import 'widgets/map_hud.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  final navigatorKey = GlobalKey<NavigatorState>();
  runApp(
    SignifyApp(
      navigatorKey: navigatorKey,
      game: MenuGame(
        onGragasPressed: () {
          navigatorKey.currentState?.push(
            MaterialPageRoute<void>(
              builder: (_) => const TrainingPage(),
            ),
          );
        },
      ),
    ),
  );
}

class SignifyApp extends StatelessWidget {
  const SignifyApp({
    super.key,
    this.game,
    this.navigatorKey,
    this.authStatus,
  });

  final MenuGame? game;
  final GlobalKey<NavigatorState>? navigatorKey;
  final Stream<bool>? authStatus;

  @override
  Widget build(BuildContext context) {
    final game = this.game;

    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'Signify',
      theme: AppTheme.light,
      home: AuthGate(
        menu: game == null
            ? const HomePage()
            : GameWidget<MenuGame>(
                game: game,
                overlayBuilderMap: {
                  'MapHud': (context, _) => const MapHud(),
                },
                initialActiveOverlays: const ['MapHud'],
              ),
        authStatus: authStatus,
      ),
    );
  }
}
