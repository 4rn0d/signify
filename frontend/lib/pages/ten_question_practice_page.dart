import 'package:flutter/material.dart';
import 'package:frontend/exercises/image_signing_exercise.dart';
import 'package:frontend/exercises/letter_signing_exercise.dart';
import 'package:frontend/exercises/multi_letter_matching_exercise.dart';
import 'package:frontend/exercises/sign_recognition_exercise.dart';
import 'package:frontend/exercises/training_exercise.dart';
import 'package:frontend/exercises/word_spelling_exercise.dart';

class TenQuestionPracticePage extends StatefulWidget {
  const TenQuestionPracticePage({super.key});

  @override
  State<TenQuestionPracticePage> createState() =>
      _TenQuestionPracticePageState();
}

class _TenQuestionPracticePageState extends State<TenQuestionPracticePage> {
  static const _questions = <TrainingExercise>[
    SignRecognitionExercise(),
    LetterSigningExercise(),
    ImageSigningExercise(),
    WordSpellingExercise(),
    MultiLetterMatchingExercise(),
    SignRecognitionExercise(),
    LetterSigningExercise(),
    ImageSigningExercise(),
    WordSpellingExercise(),
    MultiLetterMatchingExercise(),
  ];

  int _currentQuestion = 0;

  void _goToPreviousQuestion() {
    if (_currentQuestion == 0) return;
    setState(() => _currentQuestion--);
  }

  void _goToNextQuestion() {
    if (_currentQuestion == _questions.length - 1) {
      Navigator.of(context).pop();
      return;
    }
    setState(() => _currentQuestion++);
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final question = _questions[_currentQuestion];

    return Scaffold(
      appBar: AppBar(title: const Text('10-question practice')),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 760),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    'QUESTION ${_currentQuestion + 1} OF ${_questions.length}',
                    style: Theme.of(context).textTheme.labelLarge,
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: (_currentQuestion + 1) / _questions.length,
                      minHeight: 8,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    question.title,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: colors.surface,
                        border: Border.all(color: colors.outlineVariant),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: question,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      OutlinedButton(
                        onPressed: _currentQuestion == 0
                            ? null
                            : _goToPreviousQuestion,
                        child: const Text('Previous'),
                      ),
                      const Spacer(),
                      FilledButton(
                        onPressed: _goToNextQuestion,
                        child: Text(
                          _currentQuestion == _questions.length - 1
                              ? 'Finish'
                              : 'Next',
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
