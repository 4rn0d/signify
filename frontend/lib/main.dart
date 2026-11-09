import 'package:flame/game.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:frontend/game/menu_game.dart';
import 'package:frontend/pages/training_page.dart';

import 'auth_gate.dart';
import 'firebase_options.dart';

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
  const SignifyApp({super.key, this.game, this.navigatorKey});

  final MenuGame? game;
  final GlobalKey<NavigatorState>? navigatorKey;

  @override
  Widget build(BuildContext context) {
    final game = this.game;

    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'Signify',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF5B6BD5),
          surface: const Color(0xFFFFFCF6),
        ),
        scaffoldBackgroundColor: const Color(0xFFF6F3EB),
        splashFactory: InkRipple.splashFactory,
        useMaterial3: true,
      ),
      home: const AuthGate(),
    );
  }
}