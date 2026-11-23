import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/pages/focused_practice_session_page.dart';
import 'package:frontend/pages/practice_page.dart';

void main() {
  testWidgets('focused lesson starts immediately and asks ten questions', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: FocusedPracticeSessionPage(
          letters: [_letter('a'), _letter('b')],
        ),
      ),
    );

    expect(find.byType(PracticePage), findsOneWidget);
    final progressPosition = tester.getTopLeft(
      find.byKey(const ValueKey('lesson-progress')),
    );
    final confirmationPosition = tester.getTopLeft(
      find.byKey(const ValueKey('confirmation-zone')),
    );
    final confirmationZone = find.byKey(const ValueKey('confirmation-zone'));
    final confirmButton = find.byKey(const ValueKey('practice-action-button'));
    final confirmButtonPosition = tester.getTopLeft(confirmButton);
    expect(find.byType(LinearProgressIndicator), findsOneWidget);
    expect(
      find.descendant(
        of: confirmationZone,
        matching: find.byType(Divider),
      ),
      findsOneWidget,
    );
    expect(
      tester.getSize(confirmButton).width,
      lessThan(tester.getSize(confirmationZone).width),
    );
    expect(
      tester.getTopRight(confirmButton).dx,
      closeTo(tester.getTopRight(confirmationZone).dx, 0.1),
    );
    expect(
      tester.getCenter(confirmButton).dy,
      closeTo(tester.getCenter(confirmationZone).dy, 0.6),
    );

    Future<void> answerOption(
      String answer, {
      bool finish = false,
      String? feedback,
    }) async {
      await tester.tap(find.byKey(ValueKey('answer-option-$answer')));
      await tester.pumpAndSettle();
      expect(find.byKey(const ValueKey('answer-feedback')), findsNothing);
      await tester.tap(find.text('Check answer'));
      await tester.pumpAndSettle();
      expect(find.byIcon(Icons.check_circle_rounded), findsOneWidget);
      if (feedback != null) expect(find.text(feedback), findsOneWidget);
      await tester.tap(find.text(finish ? 'Finish lesson' : 'Continue'));
      await tester.pumpAndSettle();
    }

    expect(find.text('QUESTION 1 OF 10'), findsOneWidget);
    expect(find.text('Which letter is this sign?'), findsOneWidget);
    final firstSign = find.byWidgetPredicate(
      (widget) => widget is Image && widget.semanticLabel == 'Sign for A',
    );
    final signPosition = tester.getTopLeft(firstSign);
    await tester.tap(find.byKey(const ValueKey('answer-option-A')));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('answer-feedback')), findsNothing);
    await tester.tap(find.text('Check answer'));
    await tester.pumpAndSettle();
    expect(tester.getTopLeft(confirmButton), confirmButtonPosition);
    final feedback = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('confirmation-feedback-background')),
    );
    expect(
        (feedback.decoration! as BoxDecoration).color, const Color(0xFFE8F5E9));
    expect(
      tester.getCenter(find.byKey(const ValueKey('answer-feedback-text'))).dy,
      closeTo(tester.getCenter(confirmButton).dy, 0.1),
    );
    final correctContinueButton = tester.widget<FilledButton>(
      find.descendant(
        of: confirmButton,
        matching: find.byType(FilledButton),
      ),
    );
    expect(
      correctContinueButton.style?.backgroundColor?.resolve({}),
      const Color(0xFF2E7D32),
    );
    expect(tester.getTopLeft(firstSign), signPosition);
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    expect(find.text('QUESTION 2 OF 10'), findsOneWidget);
    expect(find.text('Which sign matches this letter?'), findsOneWidget);
    expect(
      tester.getTopLeft(find.byKey(const ValueKey('lesson-progress'))),
      progressPosition,
    );
    expect(
      tester.getTopLeft(find.byKey(const ValueKey('confirmation-zone'))),
      confirmationPosition,
    );
    final matchingSign = find.byWidgetPredicate(
      (widget) => widget is Image && widget.semanticLabel == 'Sign for B',
    );
    expect(tester.getSize(matchingSign).height, greaterThan(60));
    expect(
        tester.widget<Image>(matchingSign).filterQuality, FilterQuality.none);
    await answerOption('B');

    expect(find.text('QUESTION 3 OF 10'), findsOneWidget);
    expect(find.text('Type the letter for this sign.'), findsOneWidget);
    expect(
      tester.getTopLeft(find.byKey(const ValueKey('lesson-progress'))),
      progressPosition,
    );
    expect(
      tester.getTopLeft(find.byKey(const ValueKey('confirmation-zone'))),
      confirmationPosition,
    );
    final signToType = find.byWidgetPredicate(
      (widget) => widget is Image && widget.semanticLabel == 'Sign for A',
    );
    expect(tester.getSize(signToType).height, greaterThan(150));
    expect(tester.widget<Image>(signToType).filterQuality, FilterQuality.none);
    final letterInput = find.byKey(const ValueKey('letter-answer-input'));
    await tester.tap(letterInput);
    await tester.enterText(letterInput, 'a');
    await tester.pumpAndSettle();
    expect(tester.widget<TextField>(letterInput).controller?.text, 'a');
    expect(find.text('Check answer'), findsOneWidget);
    await tester.tap(find.text('Check answer'));
    await tester.pumpAndSettle();
    expect(find.text('Correct!'), findsOneWidget);
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    expect(find.text('QUESTION 4 OF 10'), findsOneWidget);
    expect(find.text('Does this sign match the letter?'), findsOneWidget);
    await answerOption('match', feedback: 'Correct!');
    await answerOption('A');
    await answerOption('B');

    await tester.enterText(
      find.byKey(const ValueKey('letter-answer-input')),
      'A',
    );
    await tester.pumpAndSettle();
    await tester.tap(find.text('Check answer'));
    await tester.pumpAndSettle();
    expect(find.text('Correct!'), findsOneWidget);
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('answer-option-match')));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('answer-feedback')), findsNothing);
    await tester.tap(find.text('Check answer'));
    await tester.pumpAndSettle();
    expect(find.textContaining('the letter is B.'), findsOneWidget);
    expect(find.byIcon(Icons.lightbulb_rounded), findsOneWidget);
    expect(
      (tester
              .widget<AnimatedContainer>(
                find.byKey(
                  const ValueKey('confirmation-feedback-background'),
                ),
              )
              .decoration! as BoxDecoration)
          .color,
      const Color(0xFFFFEBEE),
    );
    final incorrectContinueButton = tester.widget<FilledButton>(
      find.descendant(
        of: confirmButton,
        matching: find.byType(FilledButton),
      ),
    );
    expect(
      incorrectContinueButton.style?.backgroundColor?.resolve({}),
      const Color(0xFFC62828),
    );
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    await answerOption('A');
    await answerOption('B', finish: true);

    expect(find.text('Lesson complete!'), findsOneWidget);
    expect(find.text('You got 9 out of 10 right.'), findsOneWidget);
  });
}

LearningLetter _letter(String character) => LearningLetter(
      id: character,
      character: character,
      order: character.codeUnitAt(0) - 64,
      language: 'ASL',
      imagePath: 'assets/asl-signs/$character.png',
      videoPath: null,
      isActive: true,
    );
