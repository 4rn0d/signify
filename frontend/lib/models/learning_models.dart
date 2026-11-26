class LearningConfig {
  const LearningConfig({
    required this.masteryThreshold,
    required this.spellingEligibilityThreshold,
    required this.defaultBlockSize,
  });

  final int masteryThreshold;
  final int spellingEligibilityThreshold;
  final int defaultBlockSize;

  factory LearningConfig.fromMap(Map<String, dynamic> data) {
    final config = LearningConfig(
      masteryThreshold: _readInt(data, 'masteryThreshold'),
      spellingEligibilityThreshold: _readInt(
        data,
        'spellingEligibilityThreshold',
      ),
      defaultBlockSize: _readInt(data, 'defaultBlockSize'),
    );
    if (config.masteryThreshold <= 0 ||
        config.spellingEligibilityThreshold < 0 ||
        config.defaultBlockSize <= 0) {
      throw const FormatException('Learning configuration values are invalid.');
    }
    return config;
  }
}

class LearningLetter {
  const LearningLetter({
    required this.id,
    required this.character,
    required this.order,
    required this.language,
    required this.imagePath,
    required this.videoPath,
    required this.isActive,
  });

  final String id;
  final String character;
  final int order;
  final String language;
  final String imagePath;
  final String? videoPath;
  final bool isActive;

  bool matchesLanguage(String requestedLanguage) =>
      language.trim().toLowerCase() == requestedLanguage.trim().toLowerCase();

  factory LearningLetter.fromMap(String id, Map<String, dynamic> data) {
    final rawCharacter = data['character'];
    final character = rawCharacter is String
        ? rawCharacter.trim().toUpperCase()
        : rawCharacter;
    final language = data['language'];
    final media = data['media'];
    final imagePath = media is Map ? media['image'] : null;
    final videoPath = media is Map ? media['video'] : null;
    final isActive = data['isActive'];

    if (character is! String ||
        character.isEmpty ||
        language is! String ||
        imagePath is! String ||
        isActive is! bool) {
      throw FormatException('Letter "$id" has invalid fields.');
    }
    if (videoPath != null && videoPath is! String) {
      throw FormatException('Letter "$id" has an invalid video path.');
    }

    return LearningLetter(
      id: id,
      character: character,
      order: _readInt(data, 'order'),
      language: language,
      imagePath: imagePath,
      videoPath: videoPath is String ? videoPath : null,
      isActive: isActive,
    );
  }
}

class LearningWord {
  const LearningWord({
    required this.id,
    required this.text,
    required this.normalizedText,
    required this.letterIds,
    required this.language,
    required this.difficulty,
    required this.isActive,
  });

  final String id;
  final String text;
  final String normalizedText;
  final List<String> letterIds;
  final String language;
  final int difficulty;
  final bool isActive;

  factory LearningWord.fromMap(String id, Map<String, dynamic> data) {
    final text = data['text'];
    final normalizedText = data['normalizedText'];
    final rawLetterIds = data['letterIds'];
    final language = data['language'];
    final isActive = data['isActive'];

    if (text is! String ||
        normalizedText is! String ||
        rawLetterIds is! List ||
        rawLetterIds.any((id) => id is! String) ||
        language is! String ||
        isActive is! bool) {
      throw FormatException('Word "$id" has invalid fields.');
    }

    return LearningWord(
      id: id,
      text: text,
      normalizedText: normalizedText,
      letterIds: List<String>.unmodifiable(rawLetterIds.cast<String>()),
      language: language,
      difficulty: _readInt(data, 'difficulty'),
      isActive: isActive,
    );
  }

  bool isEligible({
    required Map<String, LetterProgress> progressByLetterId,
    required int spellingEligibilityThreshold,
  }) {
    return letterIds.isNotEmpty &&
        letterIds.every(
          (id) =>
              (progressByLetterId[id]?.masteryScore ?? 0) >=
              spellingEligibilityThreshold,
        );
  }
}

class LetterProgress {
  const LetterProgress({
    required this.masteryScore,
    required this.totalAttempts,
    required this.correctAttempts,
    required this.incorrectAttempts,
  });

  const LetterProgress.empty()
      : masteryScore = 0,
        totalAttempts = 0,
        correctAttempts = 0,
        incorrectAttempts = 0;

  final int masteryScore;
  final int totalAttempts;
  final int correctAttempts;
  final int incorrectAttempts;

  factory LetterProgress.fromMap(Map<String, dynamic> data) {
    final progress = LetterProgress(
      masteryScore: _readInt(data, 'masteryScore'),
      totalAttempts: _readInt(data, 'totalAttempts'),
      correctAttempts: _readInt(data, 'correctAttempts'),
      incorrectAttempts: _readInt(data, 'incorrectAttempts'),
    );
    if (progress.masteryScore < 0 ||
        progress.totalAttempts < 0 ||
        progress.correctAttempts < 0 ||
        progress.incorrectAttempts < 0) {
      throw const FormatException('Letter progress values cannot be negative.');
    }
    return progress;
  }

  LetterProgress applyResults(
    Iterable<bool> results, {
    required int masteryThreshold,
  }) {
    var score = masteryScore;
    var correct = correctAttempts;
    var incorrect = incorrectAttempts;
    var total = totalAttempts;

    for (final isCorrect in results) {
      total++;
      if (isCorrect) {
        correct++;
        if (score < masteryThreshold) score++;
      } else {
        incorrect++;
        if (score > 0) score--;
      }
    }

    return LetterProgress(
      masteryScore: score,
      totalAttempts: total,
      correctAttempts: correct,
      incorrectAttempts: incorrect,
    );
  }
}

class LetterAttemptResult {
  const LetterAttemptResult({
    required this.letterId,
    required this.isCorrect,
  });

  final String letterId;
  final bool isCorrect;

  Map<String, Object> toMap() => {
        'letterId': letterId,
        'isCorrect': isCorrect,
      };
}

class LetterMastery {
  const LetterMastery({
    required this.letter,
    required this.progress,
    required this.masteryGap,
  });

  final LearningLetter letter;
  final LetterProgress progress;
  final int masteryGap;
}

class LearningBlock {
  const LearningBlock({
    required this.letters,
    required this.activeLetterCount,
  });

  final List<LetterMastery> letters;
  final int activeLetterCount;

  factory LearningBlock.generate({
    required Iterable<LearningLetter> activeLetters,
    required Map<String, LetterProgress> progressByLetterId,
    required LearningConfig config,
  }) {
    final candidates =
        activeLetters.where((letter) => letter.isActive).toList();
    final prioritized = candidates
        .where(
      (letter) =>
          (progressByLetterId[letter.id]?.masteryScore ?? 0) <
          config.masteryThreshold,
    )
        .map((letter) {
      final progress =
          progressByLetterId[letter.id] ?? const LetterProgress.empty();
      return LetterMastery(
        letter: letter,
        progress: progress,
        masteryGap: config.masteryThreshold - progress.masteryScore,
      );
    }).toList()
      ..sort((a, b) {
        final byGap = b.masteryGap.compareTo(a.masteryGap);
        if (byGap != 0) return byGap;
        final byOrder = a.letter.order.compareTo(b.letter.order);
        return byOrder != 0 ? byOrder : a.letter.id.compareTo(b.letter.id);
      });

    return LearningBlock(
      letters: List.unmodifiable(prioritized.take(config.defaultBlockSize)),
      activeLetterCount: candidates.length,
    );
  }
}

int _readInt(Map<String, dynamic> data, String key) {
  final value = data[key];
  if (value is! num || !value.isFinite || value % 1 != 0) {
    throw FormatException('Expected "$key" to be an integer.');
  }
  return value.toInt();
}
