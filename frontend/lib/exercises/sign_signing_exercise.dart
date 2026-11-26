import 'package:flutter/material.dart';

import 'exercise.dart';
import 'exercise_widgets.dart';
import 'sign_recognition_exercise.dart';

enum SignSigningType { image, letter }

class SignSigningExercise extends Exercise {
  const SignSigningExercise({
    required this.letter,
    required this.type,
    required this.onResult,
    super.key,
  });

  final String letter;
  final SignSigningType type;
  final ValueChanged<bool> onResult;

  @override
  String get title => switch (type) {
        SignSigningType.image => 'Copy the Sign',
        SignSigningType.letter => 'Sign the Letter',
      };

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Sign the target using the camera.',
            style: Theme.of(context).textTheme.titleLarge,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 150,
            child: Center(
              child: switch (type) {
                SignSigningType.image => ExerciseSignImage(
                    character: letter,
                    width: 150,
                    height: 150,
                    semanticLabel: 'Sign for $letter',
                  ),
                SignSigningType.letter => Text(
                    letter,
                    style: Theme.of(context).textTheme.displayLarge,
                  ),
              },
            ),
          ),
          Expanded(
            child: SignRecognitionExercise(
              targetLetter: letter,
              onResult: onResult,
            ),
          ),
        ],
      );
}
