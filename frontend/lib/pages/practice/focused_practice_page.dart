import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:frontend/exercises/exercise.dart';
import 'package:frontend/exercises/multi_letter_matching_exercise.dart';
import 'package:frontend/exercises/sign_choice_exercise.dart';
import 'package:frontend/exercises/sign_signing_exercise.dart';
import 'package:frontend/exercises/word_spelling_exercise.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/pages/practice/practice_question_page.dart';
import 'package:frontend/services/auth_service.dart';
import 'package:frontend/services/learning_service.dart';

class FocusedPracticePage extends StatefulWidget {
  const FocusedPracticePage({
    required this.letters,
    this.words = const [],
    this.learningService,
    super.key,
  }) : assert(letters.length > 0);

  final List<LearningLetter> letters;
  final List<LearningWord> words;
  final LearningService? learningService;

  @override
  State<FocusedPracticePage> createState() => _FocusedPracticePageState();
}

class _FocusedPracticePageState extends State<FocusedPracticePage> {
  static const _questionCount = 10;
  static const _alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  final _random = Random();
  late final List<LearningLetter> _questions;
  late final List<LearningWord> _spellingWords;
  late final List<_ExerciseType> _exerciseTypes;
  late List<String> _choiceOptions;
  String? _selectedChoice;
  late List<String> _matchingSignLetters;
  late List<String> _matchingLetterOptions;
  late List<String?> _matchingAnswers;
  int? _selectedMatchingSignIndex;
  String? _selectedMatchingLetter;
  int? _incorrectMatchingSignIndex;
  String? _incorrectMatchingLetter;
  String? _matchingFeedback;
  Timer? _matchingErrorTimer;
  int _questionIndex = 0;
  int _score = 0;
  bool? _isCorrect;
  String? _feedback;
  bool _saving = false;
  bool _finished = false;

  LearningLetter get _currentLetter => _questions[_questionIndex];
  _ExerciseType get _exerciseType =>
      _exerciseTypes[_questionIndex % _exerciseTypes.length];
  LearningWord get _currentWord =>
      _spellingWords[_questionIndex % _spellingWords.length];

  Exercise get _currentExercise {
    final character = _currentLetter.character.trim().toUpperCase();
    return switch (_exerciseType) {
      _ExerciseType.imageSigning => SignSigningExercise(
          letter: character,
          type: SignSigningType.image,
          onResult: _onCameraResult,
        ),
      _ExerciseType.letterSigning => SignSigningExercise(
          letter: character,
          type: SignSigningType.letter,
          onResult: _onCameraResult,
        ),
      _ExerciseType.signToLetter => SignChoiceExercise(
          targetLetter: character,
          options: _choiceOptions,
          type: SignChoiceType.signToLetter,
          selectedOption: _selectedChoice,
          answered: _isCorrect != null,
          onOptionSelected: _onChoiceSelected,
        ),
      _ExerciseType.letterToSign => SignChoiceExercise(
          targetLetter: character,
          options: _choiceOptions,
          type: SignChoiceType.letterToSign,
          selectedOption: _selectedChoice,
          answered: _isCorrect != null,
          onOptionSelected: _onChoiceSelected,
        ),
      _ExerciseType.wordSpelling => WordSpellingExercise(
          word: _currentWord.text,
          letters: _wordLetters(_currentWord),
          onResult: _onWordResult,
        ),
      _ExerciseType.multiLetterMatching => MultiLetterMatchingExercise(
          signLetters: _matchingSignLetters,
          letterOptions: _matchingLetterOptions,
          selectedAnswers: _matchingAnswers,
          selectedSignIndex: _selectedMatchingSignIndex,
          selectedLetter: _selectedMatchingLetter,
          incorrectSignIndex: _incorrectMatchingSignIndex,
          incorrectLetter: _incorrectMatchingLetter,
          feedback: _matchingFeedback,
          answered: _isCorrect != null,
          onSignSelected: _onMatchingSignSelected,
          onLetterSelected: _onMatchingLetterSelected,
        ),
    };
  }

  @override
  void initState() {
    super.initState();
    _questions = List.unmodifiable(
      List.generate(
        _questionCount,
        (index) => widget.letters[index % widget.letters.length],
      ),
    );
    _spellingWords = widget.words
        .where(
          (word) =>
              _wordLetters(word).isNotEmpty &&
              _wordLetters(word).length == word.letterIds.length,
        )
        .toList(growable: false);
    _exerciseTypes = [
      _ExerciseType.imageSigning,
      _ExerciseType.letterSigning,
      _ExerciseType.signToLetter,
      _ExerciseType.letterToSign,
      if (_spellingWords.isNotEmpty) _ExerciseType.wordSpelling,
      if (widget.letters
              .map((letter) => letter.character.trim().toUpperCase())
              .toSet()
              .length >
          1)
        _ExerciseType.multiLetterMatching,
    ];
    _prepareChoiceOptions();
    if (_exerciseTypes.contains(_ExerciseType.multiLetterMatching)) {
      _prepareMatching();
    }
  }

  @override
  void dispose() {
    _matchingErrorTimer?.cancel();
    super.dispose();
  }

  List<String> _wordLetters(LearningWord word) => word.normalizedText
      .toUpperCase()
      .split('')
      .where(_alphabet.contains)
      .toList(growable: false);

  void _prepareMatching() {
    final target = _currentLetter.character.trim().toUpperCase();
    final distractors = widget.letters
        .map((letter) => letter.character.trim().toUpperCase())
        .toSet()
        .where((letter) => letter != target)
        .toList()
      ..remove(target)
      ..shuffle(_random);
    _matchingSignLetters = [target, ...distractors.take(3)]..shuffle(_random);
    _matchingLetterOptions = [..._matchingSignLetters]..shuffle(_random);
    _matchingAnswers = List<String?>.filled(_matchingSignLetters.length, null);
    _selectedMatchingSignIndex = null;
    _selectedMatchingLetter = null;
    _incorrectMatchingSignIndex = null;
    _incorrectMatchingLetter = null;
    _matchingFeedback = null;
  }

  String _feedbackMessage(bool isCorrect) {
    if (isCorrect) return 'Correct!';

    return switch (_exerciseType) {
      _ExerciseType.imageSigning ||
      _ExerciseType.letterSigning ||
      _ExerciseType.signToLetter ||
      _ExerciseType.letterToSign =>
        'Not quite. Review the sign and try again.',
      _ExerciseType.wordSpelling =>
        'Not quite. Practice spelling each letter in the word.',
      _ExerciseType.multiLetterMatching =>
        'Some signs were matched to the wrong letter.',
    };
  }

  void _onMatchingSignSelected(int index) {
    if (_isCorrect != null) return;
    final selectedLetter = _selectedMatchingLetter;
    if (selectedLetter != null) {
      unawaited(_matchPair(index, selectedLetter));
      return;
    }

    _matchingErrorTimer?.cancel();
    setState(() {
      _selectedMatchingSignIndex = index;
      _incorrectMatchingSignIndex = null;
      _incorrectMatchingLetter = null;
      _matchingFeedback = null;
    });
  }

  Future<void> _onMatchingLetterSelected(String letter) async {
    final selectedIndex = _selectedMatchingSignIndex;
    if (_isCorrect != null) return;
    if (selectedIndex != null) {
      await _matchPair(selectedIndex, letter);
      return;
    }

    _matchingErrorTimer?.cancel();
    setState(() {
      _selectedMatchingLetter = letter;
      _selectedMatchingSignIndex = null;
      _incorrectMatchingSignIndex = null;
      _incorrectMatchingLetter = null;
      _matchingFeedback = null;
    });
  }

  Future<void> _matchPair(int signIndex, String letter) async {
    _matchingErrorTimer?.cancel();
    if (_matchingSignLetters[signIndex] != letter) {
      setState(() {
        _selectedMatchingSignIndex = null;
        _selectedMatchingLetter = null;
        _incorrectMatchingSignIndex = signIndex;
        _incorrectMatchingLetter = letter;
        _matchingFeedback = 'Not a match. Try another pair.';
      });
      _matchingErrorTimer = Timer(const Duration(milliseconds: 700), () {
        if (!mounted ||
            _incorrectMatchingSignIndex != signIndex ||
            _incorrectMatchingLetter != letter) {
          return;
        }
        setState(() {
          _incorrectMatchingSignIndex = null;
          _incorrectMatchingLetter = null;
          _matchingFeedback = null;
        });
      });
      return;
    }

    final answers = [..._matchingAnswers];
    answers[signIndex] = letter;
    setState(() {
      _matchingAnswers[signIndex] = letter;
      _selectedMatchingSignIndex = null;
      _selectedMatchingLetter = null;
      _incorrectMatchingSignIndex = null;
      _incorrectMatchingLetter = null;
      _matchingFeedback = 'Correct match!';
    });
    if (answers.every((answer) => answer != null)) {
      await _selectAnswer(true);
    }
  }

  Future<void> _selectAnswer(
    bool isCorrect, {
    List<bool>? spellingResults,
  }) async {
    if (_isCorrect != null) return;
    setState(() {
      _isCorrect = isCorrect;
      _feedback = _feedbackMessage(isCorrect);
      if (isCorrect) _score++;
    });

    final uid = AuthService().currentUser?.uid;
    if (uid == null) return;

    setState(() => _saving = true);
    try {
      final learningService = widget.learningService ?? LearningService();
      switch (_exerciseType) {
        case _ExerciseType.imageSigning:
        case _ExerciseType.letterSigning:
        case _ExerciseType.signToLetter:
        case _ExerciseType.letterToSign:
          await learningService.recordLetterAttempt(
            uid: uid,
            exerciseType: 'focused_practice_${_exerciseType.name}',
            letterId: _currentLetter.id,
            isCorrect: isCorrect,
          );
        case _ExerciseType.wordSpelling:
          final results = spellingResults ??
              List<bool>.filled(_currentWord.letterIds.length, isCorrect);
          await learningService.recordSpellingAttempt(
            uid: uid,
            wordId: _currentWord.id,
            letterResults: [
              for (var index = 0;
                  index < _currentWord.letterIds.length &&
                      index < results.length;
                  index++)
                LetterAttemptResult(
                  letterId: _currentWord.letterIds[index],
                  isCorrect: results[index],
                ),
            ],
          );
        case _ExerciseType.multiLetterMatching:
          final letterIdByCharacter = {
            for (final letter in widget.letters)
              letter.character.trim().toUpperCase(): letter.id,
          };
          await learningService.recordLetterMatchingAttempt(
            uid: uid,
            letterResults: [
              for (var index = 0; index < _matchingSignLetters.length; index++)
                LetterAttemptResult(
                  letterId: letterIdByCharacter[_matchingSignLetters[index]]!,
                  isCorrect:
                      _matchingAnswers[index] == _matchingSignLetters[index],
                ),
            ],
          );
      }
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
    if (_isCorrect == null || _saving) return;
    ScaffoldMessenger.of(context).hideCurrentSnackBar();
    if (_questionIndex == _questionCount - 1) {
      setState(() => _finished = true);
      return;
    }
    setState(() {
      _questionIndex++;
      _isCorrect = null;
      _feedback = null;
      _prepareChoiceOptions();
      if (_exerciseType == _ExerciseType.multiLetterMatching) {
        _prepareMatching();
      }
    });
  }

  Future<void> _onCameraResult(bool signCompleted) async {
    await _selectAnswer(signCompleted);
  }

  void _prepareChoiceOptions() {
    final target = _currentLetter.character.trim().toUpperCase();
    final distractors = _alphabet
        .split('')
        .where((letter) => letter != target)
        .toList()
      ..shuffle(_random);
    _choiceOptions = [target, ...distractors.take(3)]..shuffle(_random);
    _selectedChoice = null;
  }

  void _onChoiceSelected(String choice) {
    if (_isCorrect != null) return;
    setState(() => _selectedChoice = choice);
    unawaited(
        _selectAnswer(choice == _currentLetter.character.trim().toUpperCase()));
  }

  Future<void> _onWordResult(List<bool> letterResults) async {
    final isCorrect =
        letterResults.length == _wordLetters(_currentWord).length &&
            letterResults.every((result) => result);
    await _selectAnswer(isCorrect, spellingResults: letterResults);
  }

  @override
  Widget build(BuildContext context) {
    if (!_finished) {
      return PracticeQuestionPage(
        title: 'Focused practice',
        questionNumber: _questionIndex + 1,
        questionCount: _questionCount,
        score: _score,
        questionContent: _currentExercise,
        answered: _isCorrect != null,
        isCorrect: _isCorrect ?? false,
        feedback: _feedback ?? '',
        saving: _saving,
        canConfirm: false,
        confirmLabel: _exerciseType == _ExerciseType.multiLetterMatching
            ? '${_matchingAnswers.where((answer) => answer != null).length} / '
                '${_matchingAnswers.length} matched'
            : 'Camera exercise',
        continueLabel: _questionIndex == _questionCount - 1
            ? 'Finish practice'
            : 'Continue',
        onContinue: _advance,
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Focused practice')),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 620),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: _PracticeComplete(score: _score),
            ),
          ),
        ),
      ),
    );
  }
}

class _PracticeComplete extends StatelessWidget {
  const _PracticeComplete({required this.score});

  final int score;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.emoji_events_outlined, size: 64),
        const SizedBox(height: 16),
        Text('Practice complete!',
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
  imageSigning,
  letterSigning,
  signToLetter,
  letterToSign,
  wordSpelling,
  multiLetterMatching,
}
