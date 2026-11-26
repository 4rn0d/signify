import 'package:flutter/material.dart';

import '../widgets/sign_prompt.dart';
import 'exercise.dart';
import 'sign_recognition_exercise.dart';

class WordSpellingExercise extends Exercise {
  const WordSpellingExercise({
    required this.word,
    required this.letters,
    required this.onResult,
    super.key,
  });

  final String word;
  final List<String> letters;
  final ValueChanged<List<bool>> onResult;

  @override
  String get title => 'Spell a Word';

  @override
  Widget build(BuildContext context) => _WordSpellingSession(
        word: word,
        letters: letters,
        onResult: onResult,
      );
}

class _WordSpellingSession extends StatefulWidget {
  const _WordSpellingSession({
    required this.word,
    required this.letters,
    required this.onResult,
  });

  final String word;
  final List<String> letters;
  final ValueChanged<List<bool>> onResult;

  @override
  State<_WordSpellingSession> createState() => _WordSpellingSessionState();
}

class _WordSpellingSessionState extends State<_WordSpellingSession> {
  final List<bool> _results = [];
  int _letterIndex = 0;
  int _cameraSession = 0;

  void _onLetterResult(bool isCorrect) {
    _results.add(isCorrect);
    if (!isCorrect || _letterIndex == widget.letters.length - 1) {
      widget.onResult(List.unmodifiable(_results));
      return;
    }

    setState(() {
      _letterIndex++;
      _cameraSession++;
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final letter = widget.letters[_letterIndex];
    return Column(
      children: [
        Text(
          'Spell this word using the camera.',
          style: Theme.of(context).textTheme.titleLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          widget.word,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 4),
        Text('Letter ${_letterIndex + 1} of ${widget.letters.length}'),
        SignPrompt(
          letter: letter,
          letterColor: colors.primary,
          borderColor: colors.outlineVariant,
          prompt: 'Sign this letter.',
        ),
        Expanded(
          child: SignRecognitionExercise(
            targetLetter: letter,
            resetToken: _cameraSession,
            onResult: _onLetterResult,
          ),
        ),
      ],
    );
  }
}
