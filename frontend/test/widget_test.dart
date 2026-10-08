import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';

void main() {
  testWidgets('shows the Signify home page', (WidgetTester tester) async {
    await tester.pumpWidget(const SignifyApp());

    expect(find.text('Signify'), findsOneWidget);
    expect(find.text('Welcome to Signify'), findsOneWidget);
    expect(find.text('Your project starts here.'), findsOneWidget);
  });
}
