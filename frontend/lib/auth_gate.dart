import 'package:flutter/material.dart';

import 'pages/auth/login_page.dart';
import 'services/auth_service.dart';

class AuthGate extends StatefulWidget {
  const AuthGate({
    super.key,
    required this.menu,
    this.authStatus,
  });

  final Widget menu;
  final Stream<bool>? authStatus;

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  late final Stream<bool> _authStatus;

  @override
  void initState() {
    super.initState();
    _authStatus = widget.authStatus ??
        AuthService().authStateChanges.map((user) => user != null);
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<bool>(
      stream: _authStatus,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
        if (snapshot.data == true) return widget.menu;
        return const LoginPage();
      },
    );
  }
}
