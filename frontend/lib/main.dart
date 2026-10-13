import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';

import 'auth_gate.dart';
import 'firebase_options.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  runApp(const SignifyApp());
}

class SignifyApp extends StatelessWidget {
  const SignifyApp({super.key});

  @override
  Widget build(BuildContext context) {
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
      home: const AuthGate(),
    );
  }
}