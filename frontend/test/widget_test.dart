import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';
import 'package:frontend/widgets/gragas_avatar.dart';

void main() {
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

  testWidgets('opens on the Signify menu and navigates to training', (
    WidgetTester tester,
  ) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const SignifyApp());

    expect(find.text('Signify'), findsOneWidget);
    expect(
      find.text('Learn sign language, one sign at a time.'),
      findsOneWidget,
    );
    expect(find.text('Start learning'), findsOneWidget);
    expect(find.text('ALPHABET PRACTICE'), findsNothing);

    await tester.tap(find.text('Start learning'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));

    expect(find.text('ALPHABET PRACTICE'), findsOneWidget);
    expect(find.text('Learn the alphabet'), findsOneWidget);
    expect(find.text('A'), findsOneWidget);
    expect(find.text('Practice the ASL sign for A.'), findsOneWidget);
    expect(find.byTooltip('Turn camera on'), findsOneWidget);
    expect(find.byType(Image), findsOneWidget);

    await tester.tap(find.text('Done'));
    await tester.pump();
    expect(find.text('Practice the ASL sign for B.'), findsOneWidget);
    expect(find.text('2 / 26'), findsOneWidget);
    expect(
      (tester.widget<Image>(find.byType(Image)).image as AssetImage).assetName,
      'assets/asl-signs/B.png',
    );

    for (var letter = 0; letter < 24; letter++) {
      await tester.tap(find.text('Done'));
      await tester.pump();
    }
    expect(find.text('Practice the ASL sign for Z.'), findsOneWidget);
    expect(find.text('26 / 26'), findsOneWidget);
    expect(
      (tester.widget<Image>(find.byType(Image)).image as AssetImage).assetName,
      'assets/asl-signs/Z.png',
    );

    await tester.tap(find.text('Done'));
    await tester.pump();
    expect(find.text('You completed the alphabet!'), findsOneWidget);
    expect(find.text('Completed'), findsOneWidget);

    debugDefaultTargetPlatformOverride = TargetPlatform.fuchsia;
    addTearDown(() {
      debugDefaultTargetPlatformOverride = null;
    });
    await tester.tap(find.text('Practice with camera (optional)'));
    await tester.pump();

    expect(
      find.text('Camera preview is not supported on this platform.'),
      findsOneWidget,
    );
    expect(find.text('Can you make this sign?'), findsOneWidget);
    expect(find.byType(Image), findsOneWidget);

    tester.view.physicalSize = const Size(1200, 800);
    await tester.pump();
    expect(find.text('Can you make this sign?'), findsOneWidget);

    debugDefaultTargetPlatformOverride = null;
    await tester.tap(find.byTooltip('Back to menu'));
    await tester.pumpAndSettle();

    expect(find.text('Start learning'), findsOneWidget);
    expect(find.text('ALPHABET PRACTICE'), findsNothing);
  });
}
