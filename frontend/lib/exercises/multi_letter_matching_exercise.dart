import 'package:flutter/material.dart';

import 'exercise.dart';
import 'exercise_widgets.dart';

class MultiLetterMatchingExercise extends Exercise {
  const MultiLetterMatchingExercise({
    required this.signLetters,
    required this.letterOptions,
    required this.selectedAnswers,
    required this.selectedSignIndex,
    required this.selectedLetter,
    required this.incorrectSignIndex,
    required this.incorrectLetter,
    required this.feedback,
    required this.answered,
    required this.onSignSelected,
    required this.onLetterSelected,
    super.key,
  });

  final List<String> signLetters;
  final List<String> letterOptions;
  final List<String?> selectedAnswers;
  final int? selectedSignIndex;
  final String? selectedLetter;
  final int? incorrectSignIndex;
  final String? incorrectLetter;
  final String? feedback;
  final bool answered;
  final ValueChanged<int> onSignSelected;
  final ValueChanged<String> onLetterSelected;

  @override
  String get title => 'Match Signs to Letters';

  @override
  Widget build(BuildContext context) {
    final selectedIndex = selectedSignIndex;
    final colors = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          'Match each sign with its letter.',
          style: Theme.of(context).textTheme.titleLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 12),
        Text(
          feedback ??
              (selectedIndex != null
                  ? 'Choose the letter for the selected sign.'
                  : selectedLetter != null
                      ? 'Choose the sign for the selected letter.'
                      : 'Select a sign or letter to start.'),
          textAlign: TextAlign.center,
          style: TextStyle(
            color: feedback == null
                ? null
                : feedback!.startsWith('Correct')
                    ? practiceCorrectColor
                    : practiceIncorrectColor,
            fontWeight: feedback == null ? null : FontWeight.w600,
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Column(
                spacing: 8,
                children: [
                  for (var index = 0; index < signLetters.length; index++)
                    Card(
                      clipBehavior: Clip.antiAlias,
                      color: _signTileColor(index, colors),
                      child: InkWell(
                        key: ValueKey('matching-sign-$index'),
                        onTap: answered || selectedAnswers[index] != null
                            ? null
                            : () => onSignSelected(index),
                        child: Container(
                          width: 72,
                          height: 72,
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: selectedIndex == index
                                  ? colors.primary
                                  : incorrectSignIndex == index
                                      ? practiceIncorrectColor
                                      : selectedAnswers[index] != null
                                          ? practiceCorrectColor
                                          : Colors.transparent,
                              width: 2,
                            ),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              ExerciseSignImage(
                                character: signLetters[index],
                                width: 72,
                                height: 72,
                                semanticLabel: 'Sign to match',
                              ),
                              if (selectedAnswers[index] case final answer?)
                                Positioned(
                                  right: 2,
                                  bottom: 2,
                                  child: DecoratedBox(
                                    decoration: BoxDecoration(
                                      color: colors.surface,
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 6,
                                        vertical: 2,
                                      ),
                                      child: Text(
                                        answer,
                                        style: Theme.of(context)
                                            .textTheme
                                            .labelLarge,
                                      ),
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(width: 24),
              Column(
                spacing: 8,
                children: [
                  for (final letter in letterOptions)
                    Card(
                      clipBehavior: Clip.antiAlias,
                      color: _letterTileColor(letter, colors),
                      child: InkWell(
                        key: ValueKey('matching-letter-$letter'),
                        onTap: answered || selectedAnswers.contains(letter)
                            ? null
                            : () => onLetterSelected(letter),
                        child: Container(
                          height: 72,
                          width: 72,
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: selectedLetter == letter
                                  ? colors.primary
                                  : incorrectLetter == letter
                                      ? practiceIncorrectColor
                                      : selectedAnswers.contains(letter)
                                          ? practiceCorrectColor
                                          : Colors.transparent,
                              width: 2,
                            ),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Center(
                            child: Text(
                              letter,
                              textAlign: TextAlign.center,
                              style: Theme.of(context).textTheme.displayMedium,
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Color? _signTileColor(int index, ColorScheme colors) {
    if (selectedSignIndex == index) return colors.secondaryContainer;
    if (incorrectSignIndex == index) {
      return practiceIncorrectColor.withValues(alpha: 0.12);
    }
    if (selectedAnswers[index] != null) {
      return practiceCorrectColor.withValues(alpha: 0.12);
    }
    return null;
  }

  Color? _letterTileColor(String letter, ColorScheme colors) {
    if (selectedLetter == letter) return colors.secondaryContainer;
    if (incorrectLetter == letter) {
      return practiceIncorrectColor.withValues(alpha: 0.12);
    }
    if (selectedAnswers.contains(letter)) {
      return practiceCorrectColor.withValues(alpha: 0.12);
    }
    return null;
  }
}
