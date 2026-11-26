import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/exercises/multi_letter_matching_exercise.dart';
import 'package:frontend/exercises/sign_choice_exercise.dart';
import 'package:frontend/exercises/sign_signing_exercise.dart';
import 'package:frontend/exercises/sign_recognition_exercise.dart';
import 'package:frontend/exercises/word_spelling_exercise.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/pages/practice/focused_practice_page.dart';
import 'package:frontend/pages/practice/practice_question_page.dart';

void main() {
  testWidgets('focused practice composes camera and matching exercises', (
    WidgetTester tester,
  ) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.linux;
    addTearDown(() {
      debugDefaultTargetPlatformOverride = null;
    });
    tester.view.physicalSize = const Size(1200, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(
        home: FocusedPracticePage(
          letters: [_letter('A'), _letter('B')],
          words: [_word('CAT')],
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byType(PracticeQuestionPage), findsOneWidget);
    expect(find.byType(SignSigningExercise), findsOneWidget);
    expect(
      tester.widget<SignSigningExercise>(find.byType(SignSigningExercise)).type,
      SignSigningType.image,
    );
    expect(find.text('Sign the target using the camera.'), findsOneWidget);
    expect(
      find.byWidgetPredicate(
        (widget) => widget is Image && widget.semanticLabel == 'Sign for A',
      ),
      findsOneWidget,
    );
    expect(find.byTooltip('Open camera'), findsOneWidget);

    Future<void> completeCameraExercise({
      int signCount = 1,
      bool finishPractice = false,
    }) async {
      for (var index = 0; index < signCount; index++) {
        await tester.tap(find.text("Can't open camera right now."));
        await tester.pumpAndSettle();
        if (signCount > 1 && index < signCount - 1) {
          expect(
              find.text('Letter ${index + 2} of $signCount'), findsOneWidget);
        }
      }
      expect(find.text('Correct!'), findsOneWidget);
      await tester.tap(
        find.text(finishPractice ? 'Finish practice' : 'Continue'),
      );
      await tester.pumpAndSettle();
    }

    Future<void> completeMatchingExercise() async {
      var exercise = tester.widget<MultiLetterMatchingExercise>(
        find.byType(MultiLetterMatchingExercise),
      );
      final wrongLetter = exercise.letterOptions.firstWhere(
        (letter) => letter != exercise.signLetters.first,
      );
      await tester.tap(find.byKey(ValueKey('matching-letter-$wrongLetter')));
      await tester.pumpAndSettle();
      exercise = tester.widget<MultiLetterMatchingExercise>(
        find.byType(MultiLetterMatchingExercise),
      );
      expect(exercise.selectedLetter, wrongLetter);
      await tester.tap(find.byKey(const ValueKey('matching-sign-0')));
      await tester.pump();
      expect(find.text('Not a match. Try another pair.'), findsOneWidget);
      exercise = tester.widget<MultiLetterMatchingExercise>(
        find.byType(MultiLetterMatchingExercise),
      );
      expect(exercise.selectedAnswers.first, isNull);
      expect(exercise.incorrectSignIndex, 0);
      expect(exercise.incorrectLetter, wrongLetter);
      await tester.pump(const Duration(milliseconds: 800));
      exercise = tester.widget<MultiLetterMatchingExercise>(
        find.byType(MultiLetterMatchingExercise),
      );
      expect(exercise.feedback, isNull);
      expect(exercise.incorrectSignIndex, isNull);

      for (var index = 0; index < exercise.signLetters.length; index++) {
        if (index.isEven) {
          await tester.tap(find.byKey(ValueKey('matching-sign-$index')));
          await tester.pumpAndSettle();
          expect(
            tester
                .widget<MultiLetterMatchingExercise>(
                  find.byType(MultiLetterMatchingExercise),
                )
                .selectedSignIndex,
            index,
          );
          await tester.tap(
            find.byKey(
              ValueKey('matching-letter-${exercise.signLetters[index]}'),
            ),
          );
        } else {
          await tester.tap(
            find.byKey(
              ValueKey('matching-letter-${exercise.signLetters[index]}'),
            ),
          );
          await tester.pumpAndSettle();
          expect(
            tester
                .widget<MultiLetterMatchingExercise>(
                  find.byType(MultiLetterMatchingExercise),
                )
                .selectedLetter,
            exercise.signLetters[index],
          );
          await tester.tap(find.byKey(ValueKey('matching-sign-$index')));
        }
        await tester.pumpAndSettle();
        final updatedExercise = tester.widget<MultiLetterMatchingExercise>(
          find.byType(MultiLetterMatchingExercise),
        );
        expect(updatedExercise.selectedAnswers[index],
            exercise.signLetters[index]);
        expect(
          tester
              .widget<InkWell>(
                find.byKey(ValueKey('matching-sign-$index')),
              )
              .onTap,
          isNull,
        );
        expect(
          tester
              .widget<InkWell>(
                find.byKey(
                  ValueKey('matching-letter-${exercise.signLetters[index]}'),
                ),
              )
              .onTap,
          isNull,
        );
      }
      expect(find.text('Correct!'), findsOneWidget);
      exercise = tester.widget<MultiLetterMatchingExercise>(
        find.byType(MultiLetterMatchingExercise),
      );
      expect(exercise.answered, isTrue);
      await tester.tap(find.text('Continue'));
      await tester.pumpAndSettle();
    }

    Future<void> completeChoiceExercise({
      required SignChoiceType type,
      bool chooseIncorrectly = false,
      bool finishPractice = false,
    }) async {
      final exercise = tester.widget<SignChoiceExercise>(
        find.byType(SignChoiceExercise),
      );
      expect(exercise.type, type);
      expect(exercise.options, hasLength(4));
      final selectedOption = chooseIncorrectly
          ? exercise.options.firstWhere(
              (option) => option != exercise.targetLetter,
            )
          : exercise.targetLetter;
      await tester.tap(
        find.byKey(ValueKey('sign-choice-$selectedOption')),
      );
      await tester.pumpAndSettle();
      expect(
        find.text(
          chooseIncorrectly
              ? 'Not quite. Review the sign and try again.'
              : 'Correct!',
        ),
        findsOneWidget,
      );
      expect(
        tester
            .widget<InkWell>(
              find.descendant(
                of: find.byKey(ValueKey('sign-choice-$selectedOption')),
                matching: find.byType(InkWell),
              ),
            )
            .onTap,
        isNull,
      );
      await tester.tap(
        find.text(finishPractice ? 'Finish practice' : 'Continue'),
      );
      await tester.pumpAndSettle();
    }

    await completeCameraExercise();
    expect(find.text('QUESTION 2 OF 10'), findsOneWidget);
    expect(find.byType(SignSigningExercise), findsOneWidget);
    expect(
      tester.widget<SignSigningExercise>(find.byType(SignSigningExercise)).type,
      SignSigningType.letter,
    );
    expect(find.text('Sign the target using the camera.'), findsOneWidget);
    expect(find.text('B'), findsOneWidget);
    await completeCameraExercise();

    expect(find.text('QUESTION 3 OF 10'), findsOneWidget);
    expect(find.byType(SignChoiceExercise), findsOneWidget);
    expect(find.text('Which letter does this sign represent?'), findsOneWidget);
    expect(
      find.byWidgetPredicate(
        (widget) => widget is Image && widget.semanticLabel == 'Sign for A',
      ),
      findsOneWidget,
    );
    await completeChoiceExercise(
      type: SignChoiceType.signToLetter,
      chooseIncorrectly: true,
    );

    expect(find.text('QUESTION 4 OF 10'), findsOneWidget);
    expect(find.text('Select the sign for this letter.'), findsOneWidget);
    expect(find.text('B'), findsOneWidget);
    expect(find.byType(Image), findsNWidgets(4));
    await completeChoiceExercise(type: SignChoiceType.letterToSign);

    expect(find.text('QUESTION 5 OF 10'), findsOneWidget);
    expect(find.byType(WordSpellingExercise), findsOneWidget);
    expect(find.text('CAT'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
    expect(find.byType(SignRecognitionExercise), findsOneWidget);
    expect(find.text('Letter 1 of 3'), findsOneWidget);
    await completeCameraExercise(signCount: 3);

    expect(find.text('QUESTION 6 OF 10'), findsOneWidget);
    expect(find.byType(MultiLetterMatchingExercise), findsOneWidget);
    expect(find.text('Match each sign with its letter.'), findsOneWidget);
    await completeMatchingExercise();

    await completeCameraExercise();
    await completeCameraExercise();
    await completeChoiceExercise(type: SignChoiceType.signToLetter);
    await completeChoiceExercise(
      type: SignChoiceType.letterToSign,
      finishPractice: true,
    );

    expect(find.text('Practice complete!'), findsOneWidget);
    expect(find.text('You got 9 out of 10 right.'), findsOneWidget);
    debugDefaultTargetPlatformOverride = null;
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

LearningWord _word(String text) => LearningWord(
      id: text.toLowerCase(),
      text: text,
      normalizedText: text,
      letterIds: text.split(''),
      language: 'ASL',
      difficulty: 1,
      isActive: true,
    );
