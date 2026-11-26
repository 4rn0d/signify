import 'package:flutter/material.dart';

import 'exercise.dart';
import 'exercise_widgets.dart';

enum SignChoiceType { signToLetter, letterToSign }

class SignChoiceExercise extends Exercise {
  const SignChoiceExercise({
    required this.targetLetter,
    required this.options,
    required this.type,
    required this.selectedOption,
    required this.answered,
    required this.onOptionSelected,
    super.key,
  }) : assert(options.length == 4);

  final String targetLetter;
  final List<String> options;
  final SignChoiceType type;
  final String? selectedOption;
  final bool answered;
  final ValueChanged<String> onOptionSelected;

  @override
  String get title => switch (type) {
        SignChoiceType.signToLetter => 'Identify the Sign',
        SignChoiceType.letterToSign => 'Find the Sign',
      };

  @override
  Widget build(BuildContext context) {
    final isSignToLetter = type == SignChoiceType.signToLetter;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          isSignToLetter
              ? 'Which letter does this sign represent?'
              : 'Select the sign for this letter.',
          style: Theme.of(context).textTheme.titleLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 150,
          child: Center(
            child: isSignToLetter
                ? ExerciseSignImage(
                    character: targetLetter,
                    width: 150,
                    height: 150,
                    semanticLabel: 'Sign for $targetLetter',
                  )
                : Text(
                    targetLetter,
                    style: Theme.of(context).textTheme.displayLarge,
                  ),
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: GridView.count(
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
            childAspectRatio: 2.75,
            children: [
              for (final option in options)
                _ChoiceTile(
                  key: ValueKey('sign-choice-$option'),
                  option: option,
                  showSignImage: !isSignToLetter,
                  selected: selectedOption == option,
                  isCorrect: option == targetLetter,
                  answered: answered,
                  onTap: () => onOptionSelected(option),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ChoiceTile extends StatelessWidget {
  const _ChoiceTile({
    required this.option,
    required this.showSignImage,
    required this.selected,
    required this.isCorrect,
    required this.answered,
    required this.onTap,
    super.key,
  });

  final String option;
  final bool showSignImage;
  final bool selected;
  final bool isCorrect;
  final bool answered;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final selectedColor =
        isCorrect ? practiceCorrectColor : practiceIncorrectColor;

    return Card(
      clipBehavior: Clip.antiAlias,
      color: selected ? selectedColor.withValues(alpha: 0.12) : null,
      child: InkWell(
        onTap: answered ? null : onTap,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: selected ? selectedColor : Colors.transparent,
              width: 2,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Center(
            child: showSignImage
                ? ExerciseSignImage(
                    character: option,
                    width: 96,
                    height: 96,
                    semanticLabel: 'Sign for $option',
                  )
                : Text(
                    option,
                    style: Theme.of(context).textTheme.displayMedium,
                  ),
          ),
        ),
      ),
    );
  }
}
