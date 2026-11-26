import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/pages/profile_page.dart';
import 'package:frontend/services/user_profile_service.dart';

void main() {
  testWidgets('shows the profile with real (empty) stats', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: ProfilePage(
          email: 'person@example.com',
          profile: Future.value(
            const UserProfile(displayName: 'Player One', exp: 120),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Profile'), findsOneWidget);
    expect(find.text('Player One'), findsWidgets);
    expect(find.text('person@example.com'), findsOneWidget);
    expect(find.text('120'), findsOneWidget);
    expect(find.text('20 / 100 XP'), findsOneWidget);
    expect(find.text('Exercises done'), findsOneWidget);
    expect(find.text('Time learning'), findsNothing);
    expect(find.byIcon(Icons.person), findsOneWidget);
  });

  testWidgets('shows a loading indicator while the profile loads', (
    WidgetTester tester,
  ) async {
    final profile = Completer<UserProfile>();
    await tester.pumpWidget(
      MaterialApp(home: ProfilePage(email: null, profile: profile.future)),
    );
    await tester.pump();

    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    expect(find.text('No email available'), findsNothing);
  });
}
