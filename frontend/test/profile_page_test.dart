import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/pages/profile_page.dart';
import 'package:frontend/services/user_profile_service.dart';

void main() {
  testWidgets('shows the user email and a placeholder profile picture', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: ProfilePage(
          email: 'person@example.com',
          profile: Future.value(
            const UserProfile(displayName: 'Player One', exp: 0),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Profile'), findsOneWidget);
    expect(find.text('Player One'), findsOneWidget);
    expect(find.text('person@example.com'), findsOneWidget);
    expect(find.text('0 EXP'), findsOneWidget);
    expect(find.byType(CircleAvatar), findsOneWidget);
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
