import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import 'package:frontend/game/menu_game.dart';

import 'pages/home_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(SignifyApp(game: MenuGame()));
}

class SignifyApp extends StatelessWidget {
  const SignifyApp({super.key, this.game});

  final MenuGame? game;

  @override
  Widget build(BuildContext context) {
    final game = this.game;

    return MaterialApp(
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
      home: game == null ? const HomePage() : GameWidget(game: game),
    );
  }
}
