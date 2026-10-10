import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';

void main() {
  testWidgets('opens on the Signify menu and navigates to training', (
    WidgetTester tester,
  ) async {
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

    await tester.tap(find.byTooltip('Back to menu'));
    await tester.pumpAndSettle();

    expect(find.text('Start learning'), findsOneWidget);
    expect(find.text('ALPHABET PRACTICE'), findsNothing);
  });
}
