import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/services/profile_stats_service.dart';

ActivityRecord _at(DateTime time, {bool correct = true, int xp = 10}) =>
    ActivityRecord(createdAt: time, isCorrect: correct, xpEarned: xp);

LetterProgress _progress(int score) => LetterProgress(
      masteryScore: score,
      totalAttempts: score,
      correctAttempts: score,
      incorrectAttempts: 0,
    );

void main() {
  // Wednesday 2026-11-25 at noon.
  final now = DateTime(2026, 11, 25, 12);

  test('empty activity gives zeroed stats', () {
    final stats = ProfileStats.compute(
      activity: const [],
      progress: const [],
      masteryThreshold: 5,
      now: now,
    );
    expect(stats.currentStreak, 0);
    expect(stats.longestStreak, 0);
    expect(stats.exercisesCompleted, 0);
    expect(stats.quizAccuracy, isNull);
    expect(stats.weeklyXp, 0);
    expect(stats.activeDays, isEmpty);
  });

  test('computes streaks, accuracy, weekly XP and mastered signs', () {
    final stats = ProfileStats.compute(
      activity: [
        // Old 4-day run in October.
        for (var d = 1; d <= 4; d++) _at(DateTime(2026, 10, d)),
        // Current run: Nov 23 (Mon) - Nov 24 (Tue), today not yet practiced.
        _at(DateTime(2026, 11, 22, 9), xp: 10), // Sunday, last week
        _at(DateTime(2026, 11, 23, 9), correct: false, xp: 0),
        _at(DateTime(2026, 11, 24, 9)),
        _at(DateTime(2026, 11, 24, 18)),
      ],
      progress: [_progress(5), _progress(5), _progress(2)],
      masteryThreshold: 5,
      now: now,
    );

    expect(stats.currentStreak, 3); // 22, 23, 24 - alive until end of today
    expect(stats.longestStreak, 4);
    expect(stats.exercisesCompleted, 8);
    expect(stats.quizAccuracy, closeTo(7 / 8, 1e-9));
    expect(stats.weeklyXp, 20); // Nov 23 onward
    expect(stats.signsLearned, 2);
    expect(stats.activeDays, {22, 23, 24});
  });

  test('streak breaks after a missed day', () {
    final stats = ProfileStats.compute(
      activity: [_at(DateTime(2026, 11, 23))],
      progress: const [],
      masteryThreshold: 5,
      now: now,
    );
    expect(stats.currentStreak, 0);
    expect(stats.longestStreak, 1);
  });
}
