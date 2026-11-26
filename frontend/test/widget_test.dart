import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/auth_gate.dart';
import 'package:frontend/main.dart';
import 'package:frontend/widgets/gragas_avatar.dart';

void main() {
  testWidgets('AuthGate routes authenticated users to the menu', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: AuthGate(
          menu: const Text('Café menu'),
          authStatus: Stream<bool>.value(true),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Café menu'), findsOneWidget);
    expect(find.text('Connexion'), findsNothing);
  });

  testWidgets('GragasAvatar switches between available animations', (
    WidgetTester tester,
  ) async {
    Widget avatar(GragasAnimation animation) => MaterialApp(
          home: SizedBox(
            width: 320,
            height: 320,
            child: GragasAvatar(animation: animation),
          ),
        );

    await tester.pumpWidget(avatar(GragasAnimation.idle));
    await tester.pump(const Duration(milliseconds: 100));

    for (final animation in GragasAnimation.values.skip(1)) {
      await tester.pumpWidget(avatar(animation));
      await tester.pump(const Duration(milliseconds: 100));
      expect(tester.takeException(), isNull);
    }
  });

  testWidgets('opens on the Signify menu and navigates to practice', (
    WidgetTester tester,
  ) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      SignifyApp(authStatus: Stream<bool>.value(true)),
    );
    await tester.pumpAndSettle();

    expect(find.text('Signify'), findsOneWidget);
    expect(
      find.text('Learn sign language, one sign at a time.'),
      findsOneWidget,
    );
    expect(find.text('Start practicing'), findsOneWidget);
    expect(find.text('ENDLESS PRACTICE'), findsNothing);

    await tester.tap(find.text('Start practicing'));
    await tester.pumpAndSettle();

    expect(find.text('Choose your practice'), findsNothing);
    expect(find.text('Focused practice'), findsOneWidget);
    expect(
      find.text('Sign in to generate a personal learning block.'),
      findsOneWidget,
    );
    expect(find.text('10 Questions'), findsNothing);
    expect(find.text('ENDLESS PRACTICE'), findsNothing);
  });
}
