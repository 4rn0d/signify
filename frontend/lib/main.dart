import 'package:flutter/material.dart';

import 'pages/home_page.dart';

void main() {
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
      home: const HomePage(),
    );
  }
}
