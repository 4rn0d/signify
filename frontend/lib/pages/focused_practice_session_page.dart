import 'dart:math';

import 'package:flutter/material.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/pages/practice_page.dart';
import 'package:frontend/services/auth_service.dart';
import 'package:frontend/services/learning_service.dart';

Widget _signImage({
  required String character,
  required double width,
  required double height,
  String? semanticLabel,
}) {
  return SizedBox(
    width: width,
    height: height,
    child: Image.asset(
      'assets/asl-signs/$character.png',
      fit: BoxFit.contain,
      filterQuality: FilterQuality.none,
      semanticLabel: semanticLabel,
    ),
  );
}

class FocusedPracticeSessionPage extends StatefulWidget {
  const FocusedPracticeSessionPage({
    required this.letters,
    this.learningService,
    super.key,
  }) : assert(letters.length > 0);

  final List<LearningLetter> letters;
  final LearningService? learningService;

  @override
  State<FocusedPracticeSessionPage> createState() =>
      _FocusedPracticeSessionPageState();
}

class _FocusedPracticeSessionPageState
    extends State<FocusedPracticeSessionPage> {
  static const _questionCount = 10;
  static const _alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  final _random = Random();
  late final List<LearningLetter> _questions;
  late List<String> _options;
  int _questionIndex = 0;
  int _score = 0;
  String? _pendingAnswer;
  String? _selectedAnswer;
  String? _feedback;
  final _answerController = TextEditingController();
  bool _saving = false;
  bool _finished = false;

  LearningLetter get _currentLetter => _questions[_questionIndex];
  _ExerciseType get _exerciseType =>
      _ExerciseType.values[_questionIndex % _ExerciseType.values.length];
  String get _correctAnswer {
    final character = _currentLetter.character.trim().toUpperCase();
    return switch (_exerciseType) {
      _ExerciseType.signToLetter || _ExerciseType.letterToSign => character,
      _ExerciseType.typeLetter => character,
      _ExerciseType.matchPair =>
        _pairedSign == character ? 'match' : 'different',
    };
  }

  bool get _pairMatches => _questionIndex % 8 == 3;
  String get _pairedSign => _pairMatches
      ? _currentLetter.character.trim().toUpperCase()
      : _options.firstWhere(
          (option) => option != _currentLetter.character.trim().toUpperCase(),
        );

  @override
  void initState() {
    super.initState();
    _questions = List.unmodifiable(
      List.generate(
        _questionCount,
        (index) => widget.letters[index % widget.letters.length],
      ),
    );
    _options = _optionsFor(_currentLetter.character);
  }

  @override
  void dispose() {
    _answerController.dispose();
    super.dispose();
  }

  List<String> _optionsFor(String correctLetter) {
    final normalizedLetter = correctLetter.trim().toUpperCase();
    final distractors = _alphabet
        .split('')
        .where((letter) => letter != normalizedLetter)
        .toList()
      ..shuffle(_random);
    return [normalizedLetter, ...distractors.take(3)]..shuffle(_random);
  }

  String _feedbackMessage({
    required _ExerciseType exerciseType,
    required String character,
    required String pairedSign,
    required bool isCorrect,
  }) {
    if (isCorrect) return 'Correct!';

    return switch (exerciseType) {
      _ExerciseType.signToLetter =>
        'Not quite. The correct letter is $character.',
      _ExerciseType.letterToSign =>
        'Not quite. Choose the sign for $character.',
      _ExerciseType.typeLetter => 'Not quite. Type $character.',
      _ExerciseType.matchPair => pairedSign == character
          ? 'They do match — the sign is $character.'
          : 'No match. The sign is $pairedSign; the letter is $character.',
    };
  }

  void _chooseAnswer(String answer) {
    if (_selectedAnswer != null) return;
    setState(() => _pendingAnswer = answer);
  }

  Future<void> _confirmAnswer() async {
    final answer = _exerciseType == _ExerciseType.typeLetter
        ? _answerController.text
        : _pendingAnswer;
    if (answer == null || answer.trim().isEmpty) return;
    await _selectAnswer(answer);
  }

  Future<void> _selectAnswer(String answer) async {
    if (_selectedAnswer != null) return;
    final letter = _currentLetter;
    final exerciseType = _exerciseType;
    final normalizedAnswer = exerciseType == _ExerciseType.matchPair
        ? answer
        : answer.trim().toUpperCase();
    final expectedLetter = letter.character.trim().toUpperCase();
    final isCorrect = switch (exerciseType) {
      _ExerciseType.signToLetter ||
      _ExerciseType.letterToSign =>
        normalizedAnswer == expectedLetter,
      _ExerciseType.typeLetter => normalizedAnswer == expectedLetter,
      _ExerciseType.matchPair =>
        (normalizedAnswer == 'match') == (_pairedSign == expectedLetter),
    };
    final feedback = _feedbackMessage(
      exerciseType: exerciseType,
      character: expectedLetter,
      pairedSign: _pairedSign,
      isCorrect: isCorrect,
    );
    setState(() {
      _pendingAnswer = null;
      _selectedAnswer = normalizedAnswer;
      _feedback = feedback;
      if (isCorrect) _score++;
    });

    final uid = AuthService().currentUser?.uid;
    if (uid == null) return;

    setState(() => _saving = true);
    try {
      await (widget.learningService ?? LearningService()).recordLetterAttempt(
        uid: uid,
        exerciseType: 'focused_practice_${exerciseType.name}',
        letterId: letter.id,
        isCorrect: isCorrect,
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save practice progress: $error')),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  void _advance() {
    if (_selectedAnswer == null || _saving) return;
    ScaffoldMessenger.of(context).hideCurrentSnackBar();
    if (_questionIndex == _questionCount - 1) {
      setState(() => _finished = true);
      return;
    }
    setState(() {
      _pendingAnswer = null;
      _questionIndex++;
      _selectedAnswer = null;
      _feedback = null;
      _answerController.clear();
      _options = _optionsFor(_currentLetter.character);
    });
  }

  void _onTypedAnswerChanged(String _) {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (!_finished) {
      return PracticePage(
        title: 'Focused lesson',
        questionNumber: _questionIndex + 1,
        questionCount: _questionCount,
        score: _score,
        questionContent: _QuestionContent(
          letter: _currentLetter,
          exerciseType: _exerciseType,
          pairedSign: _pairedSign,
          options: _options,
          pendingAnswer: _pendingAnswer,
          selectedAnswer: _selectedAnswer,
          correctAnswer: _correctAnswer,
          answerController: _answerController,
          onChoose: _chooseAnswer,
          onTypedAnswerChanged: _onTypedAnswerChanged,
        ),
        answered: _selectedAnswer != null,
        isCorrect: _selectedAnswer == _correctAnswer,
        feedback: _feedback ?? '',
        saving: _saving,
        canConfirm: _exerciseType == _ExerciseType.typeLetter
            ? _answerController.text.trim().isNotEmpty
            : _pendingAnswer != null,
        onConfirm: _confirmAnswer,
        continueLabel:
            _questionIndex == _questionCount - 1 ? 'Finish lesson' : 'Continue',
        onContinue: _advance,
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Focused lesson')),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 620),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: _LessonComplete(score: _score),
            ),
          ),
        ),
      ),
    );
  }
}

class _QuestionContent extends StatelessWidget {
  const _QuestionContent({
    required this.letter,
    required this.exerciseType,
    required this.pairedSign,
    required this.options,
    required this.pendingAnswer,
    required this.selectedAnswer,
    required this.correctAnswer,
    required this.answerController,
    required this.onChoose,
    required this.onTypedAnswerChanged,
  });

  final LearningLetter letter;
  final _ExerciseType exerciseType;
  final String pairedSign;
  final List<String> options;
  final String? pendingAnswer;
  final String? selectedAnswer;
  final String correctAnswer;
  final TextEditingController answerController;
  final ValueChanged<String> onChoose;
  final ValueChanged<String> onTypedAnswerChanged;

  @override
  Widget build(BuildContext context) {
    final answered = selectedAnswer != null;
    final character = letter.character.trim().toUpperCase();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          switch (exerciseType) {
            _ExerciseType.signToLetter => 'Which letter is this sign?',
            _ExerciseType.letterToSign => 'Which sign matches this letter?',
            _ExerciseType.typeLetter => 'Type the letter for this sign.',
            _ExerciseType.matchPair => 'Does this sign match the letter?',
          },
          style: Theme.of(context).textTheme.titleLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Expanded(
          child: Center(
            child: switch (exerciseType) {
              _ExerciseType.signToLetter => _signImage(
                  character: character,
                  width: 180,
                  height: 180,
                  semanticLabel: 'Sign for $character',
                ),
              _ExerciseType.typeLetter => _signImage(
                  character: character,
                  width: 220,
                  height: 220,
                  semanticLabel: 'Sign for $character',
                ),
              _ExerciseType.letterToSign => Text(
                  character,
                  style: Theme.of(context).textTheme.displayLarge,
                ),
              _ExerciseType.matchPair => Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      character,
                      style: Theme.of(context).textTheme.displayLarge,
                    ),
                    const SizedBox(width: 24),
                    _signImage(
                      character: pairedSign,
                      width: 160,
                      height: 160,
                      semanticLabel: 'Sign to compare',
                    ),
                  ],
                ),
            },
          ),
        ),
        const SizedBox(height: 16),
        _AnswerChoices(
          exerciseType: exerciseType,
          options: options,
          pendingAnswer: pendingAnswer,
          selectedAnswer: selectedAnswer,
          correctAnswer: correctAnswer,
          answerController: answerController,
          answered: answered,
          onChoose: onChoose,
          onTypedAnswerChanged: onTypedAnswerChanged,
        ),
      ],
    );
  }
}

class _AnswerChoices extends StatelessWidget {
  const _AnswerChoices({
    required this.exerciseType,
    required this.options,
    required this.pendingAnswer,
    required this.selectedAnswer,
    required this.correctAnswer,
    required this.answerController,
    required this.answered,
    required this.onChoose,
    required this.onTypedAnswerChanged,
  });

  final _ExerciseType exerciseType;
  final List<String> options;
  final String? pendingAnswer;
  final String? selectedAnswer;
  final String correctAnswer;
  final TextEditingController answerController;
  final bool answered;
  final ValueChanged<String> onChoose;
  final ValueChanged<String> onTypedAnswerChanged;

  @override
  Widget build(BuildContext context) {
    if (exerciseType == _ExerciseType.typeLetter) {
      return TextField(
        key: const ValueKey('letter-answer-input'),
        controller: answerController,
        enabled: !answered,
        maxLength: 1,
        textCapitalization: TextCapitalization.characters,
        textAlign: TextAlign.center,
        onChanged: onTypedAnswerChanged,
        decoration: const InputDecoration(
          hintText: 'Letter',
          counterText: '',
          border: OutlineInputBorder(),
        ),
      );
    }

    final colors = Theme.of(context).colorScheme;
    final answerOptions = exerciseType == _ExerciseType.matchPair
        ? const ['match', 'different']
        : options;
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      shrinkWrap: true,
      mainAxisExtent: exerciseType == _ExerciseType.letterToSign ? 94 : null,
      childAspectRatio: 2.8,
      physics: const NeverScrollableScrollPhysics(),
      children: [
        for (final option in answerOptions)
          Builder(
            builder: (context) {
              final isSelected =
                  (answered ? selectedAnswer : pendingAnswer) == option;
              final isCorrectOption = option == correctAnswer;
              return AnimatedScale(
                duration: const Duration(milliseconds: 180),
                scale: answered && isCorrectOption
                    ? 1.04
                    : answered && isSelected
                        ? 0.97
                        : 1,
                child: OutlinedButton(
                  key: ValueKey('answer-option-$option'),
                  onPressed: answered ? null : () => onChoose(option),
                  style: OutlinedButton.styleFrom(
                    backgroundColor: answered
                        ? isCorrectOption
                            ? practiceCorrectBackgroundColor
                            : isSelected
                                ? practiceIncorrectBackgroundColor
                                : null
                        : isSelected
                            ? colors.secondaryContainer
                            : null,
                    side: answered && isCorrectOption
                        ? const BorderSide(
                            color: practiceCorrectColor,
                            width: 2,
                          )
                        : answered && isSelected
                            ? const BorderSide(
                                color: practiceIncorrectColor,
                                width: 2,
                              )
                            : null,
                  ),
                  child: switch (exerciseType) {
                    _ExerciseType.letterToSign => SizedBox(
                        width: 128,
                        height: 72,
                        child: Image.asset(
                          'assets/asl-signs/$option.png',
                          fit: BoxFit.contain,
                          filterQuality: FilterQuality.none,
                          semanticLabel: 'Sign for $option',
                        ),
                      ),
                    _ExerciseType.matchPair => Text(
                        option == 'match' ? 'They match' : 'They do not match',
                        textAlign: TextAlign.center,
                      ),
                    _ => Text(
                        option,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                  },
                ),
              );
            },
          ),
      ],
    );
  }
}

class _LessonComplete extends StatelessWidget {
  const _LessonComplete({required this.score});

  final int score;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.emoji_events_outlined, size: 64),
        const SizedBox(height: 16),
        Text('Lesson complete!',
            style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 8),
        Text('You got $score out of 10 right.'),
        const SizedBox(height: 24),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Done'),
        ),
      ],
    );
  }
}

enum _ExerciseType {
  signToLetter,
  letterToSign,
  typeLetter,
  matchPair,
}
