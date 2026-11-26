import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/models/learning_models.dart';

void main() {
  const config = LearningConfig(
    masteryThreshold: 10,
    spellingEligibilityThreshold: 5,
    defaultBlockSize: 2,
  );

  test('learning block treats missing progress as zero and prioritizes gaps',
      () {
    final block = LearningBlock.generate(
      activeLetters: [
        _letter('A', order: 1),
        _letter('B', order: 2),
        _letter('C', order: 3),
        _letter('D', order: 4),
        _letter('E', order: 5, isActive: false),
      ],
      progressByLetterId: {
        'B': _progress(9),
        'C': _progress(10),
        'D': _progress(5),
      },
      config: config,
    );

    expect(
      block.letters.map((entry) => entry.letter.id),
      ['A', 'D'],
    );
    expect(block.letters.first.progress.masteryScore, 0);
    expect(block.letters.first.masteryGap, 10);
    expect(block.activeLetterCount, 4);
  });

  test('an empty block distinguishes missing active letters from mastery', () {
    final noLetters = LearningBlock.generate(
      activeLetters: const [],
      progressByLetterId: const {},
      config: config,
    );
    final mastered = LearningBlock.generate(
      activeLetters: [_letter('A', order: 1)],
      progressByLetterId: {'A': _progress(config.masteryThreshold)},
      config: config,
    );

    expect(noLetters.letters, isEmpty);
    expect(noLetters.activeLetterCount, 0);
    expect(mastered.letters, isEmpty);
    expect(mastered.activeLetterCount, 1);
  });

  test('language matching ignores case and surrounding whitespace', () {
    expect(_letter('A', order: 1).matchesLanguage(' asl '), isTrue);
  });

  test('letter parsing normalizes characters for consistent matching', () {
    final letter = LearningLetter.fromMap('letter-a', {
      'character': ' a ',
      'order': 1,
      'language': 'ASL',
      'media': {'image': 'assets/asl-signs/A.png'},
      'isActive': true,
    });

    expect(letter.character, 'A');
  });

  test('progress increments correct attempts and penalizes incorrect attempts',
      () {
    final updated = _progress(1).applyResults(
      [true, false, false, true],
      masteryThreshold: 3,
    );

    expect(updated.masteryScore, 1);
    expect(updated.totalAttempts, 4);
    expect(updated.correctAttempts, 2);
    expect(updated.incorrectAttempts, 2);
  });

  test('mastery score stays between zero and threshold', () {
    final lowered = _progress(0).applyResults(
      [false],
      masteryThreshold: 10,
    );
    final raised = _progress(10).applyResults(
      [true],
      masteryThreshold: 10,
    );

    expect(lowered.masteryScore, 0);
    expect(raised.masteryScore, 10);
  });

  test('spelling words require every letter to meet the threshold', () {
    const word = LearningWord(
      id: 'word_chat',
      text: 'CHAT',
      normalizedText: 'CHAT',
      letterIds: ['C', 'H', 'A', 'T'],
      language: 'fr',
      difficulty: 1,
      isActive: true,
    );
    final eligibleProgress = {
      for (final id in word.letterIds) id: _progress(5),
    };

    expect(
      word.isEligible(
        progressByLetterId: eligibleProgress,
        spellingEligibilityThreshold: 5,
      ),
      isTrue,
    );
    expect(
      word.isEligible(
        progressByLetterId: {...eligibleProgress, 'T': _progress(4)},
        spellingEligibilityThreshold: 5,
      ),
      isFalse,
    );
    expect(
      word.isEligible(
        progressByLetterId: {...eligibleProgress}..remove('T'),
        spellingEligibilityThreshold: 5,
      ),
      isFalse,
    );
  });
}

LearningLetter _letter(
  String id, {
  required int order,
  bool isActive = true,
}) =>
    LearningLetter(
      id: id,
      character: id,
      order: order,
      language: 'ASL',
      imagePath: 'letters/$id/image.png',
      videoPath: null,
      isActive: isActive,
    );

LetterProgress _progress(int masteryScore) => LetterProgress(
      masteryScore: masteryScore,
      totalAttempts: 0,
      correctAttempts: 0,
      incorrectAttempts: 0,
    );
