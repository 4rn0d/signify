import 'package:cloud_firestore/cloud_firestore.dart';

import '../models/learning_models.dart';

class LevelInfo {
  const LevelInfo({
    required this.level,
    required this.title,
    required this.xpIntoLevel,
    required this.xpForLevel,
  });

  final int level;
  final String title;
  final int xpIntoLevel;
  final int xpForLevel;

  double get progress => xpForLevel == 0 ? 0 : xpIntoLevel / xpForLevel;
  int get xpToNext => xpForLevel - xpIntoLevel;
}

/// One recorded exercise, as stored in users/{uid}/attempts.
class ActivityRecord {
  const ActivityRecord({
    required this.createdAt,
    required this.isCorrect,
    required this.xpEarned,
  });

  final DateTime createdAt;
  final bool isCorrect;
  final int xpEarned;

  /// Returns null for attempts that are still pending a server timestamp or
  /// are malformed, so one bad document does not break the profile page.
  static ActivityRecord? fromMap(Map<String, dynamic> data) {
    final createdAt = data['createdAt'];
    if (createdAt is! Timestamp) return null;

    // Attempts written before XP existed have no top-level isCorrect for
    // spelling/matching; derive it from the per-letter results instead.
    var isCorrect = data['isCorrect'];
    if (isCorrect is! bool) {
      final results = data['letterResults'];
      isCorrect = results is List &&
          results.isNotEmpty &&
          results.every((r) => r is Map && r['isCorrect'] == true);
    }

    final xp = data['xpEarned'];
    return ActivityRecord(
      createdAt: createdAt.toDate(),
      isCorrect: isCorrect,
      xpEarned: xp is num ? xp.toInt() : 0,
    );
  }
}

class ProfileStats {
  const ProfileStats({
    required this.currentStreak,
    required this.longestStreak,
    required this.signsLearned,
    required this.exercisesCompleted,
    required this.quizAccuracy,
    required this.weeklyXp,
    required this.calendarMonth,
    required this.activeDays,
  });

  final int currentStreak;
  final int longestStreak;

  /// Letters whose mastery score has reached the mastery threshold.
  final int signsLearned;
  final int exercisesCompleted;

  /// 0.0 - 1.0, or null when there are no attempts yet.
  final double? quizAccuracy;

  /// XP earned since Monday 00:00 (local time).
  final int weeklyXp;

  /// Month shown in the streak calendar (day is always 1).
  final DateTime calendarMonth;

  /// Days of [calendarMonth] (1-31) on which the user practiced.
  final Set<int> activeDays;

  factory ProfileStats.compute({
    required Iterable<ActivityRecord> activity,
    required Iterable<LetterProgress> progress,
    required int masteryThreshold,
    required DateTime now,
  }) {
    final records = activity.toList();
    final today = _dateOnly(now);
    final weekStart = today.subtract(Duration(days: today.weekday - 1));

    final practicedDays = <DateTime>{
      for (final record in records) _dateOnly(record.createdAt),
    };
    final sortedDays = practicedDays.toList()..sort();

    var longest = 0;
    var run = 0;
    DateTime? previous;
    for (final day in sortedDays) {
      run = previous != null && _daysBetween(previous, day) == 1 ? run + 1 : 1;
      if (run > longest) longest = run;
      previous = day;
    }

    // The streak stays alive until the end of today, so a streak that ended
    // yesterday still counts.
    var current = 0;
    var cursor = practicedDays.contains(today)
        ? today
        : today.subtract(const Duration(days: 1));
    while (practicedDays.contains(cursor)) {
      current++;
      cursor = DateTime(cursor.year, cursor.month, cursor.day - 1);
    }

    final correct = records.where((record) => record.isCorrect).length;

    return ProfileStats(
      currentStreak: current,
      longestStreak: longest,
      signsLearned: progress
          .where((letter) => letter.masteryScore >= masteryThreshold)
          .length,
      exercisesCompleted: records.length,
      quizAccuracy: records.isEmpty ? null : correct / records.length,
      weeklyXp: records
          .where((record) => !record.createdAt.isBefore(weekStart))
          .fold(0, (total, record) => total + record.xpEarned),
      calendarMonth: DateTime(now.year, now.month),
      activeDays: {
        for (final day in practicedDays)
          if (day.year == now.year && day.month == now.month) day.day,
      },
    );
  }

  static DateTime _dateOnly(DateTime time) {
    final local = time.toLocal();
    return DateTime(local.year, local.month, local.day);
  }

  // Rounded so daylight-saving shifts do not turn one day into 23h.
  static int _daysBetween(DateTime a, DateTime b) =>
      (b.difference(a).inHours / 24).round();
}

class LeaderboardEntry {
  const LeaderboardEntry({
    required this.name,
    required this.weeklyXp,
    this.isCurrentUser = false,
  });

  final String name;
  final int weeklyXp;
  final bool isCurrentUser;
}

class ProfileStatsService {
  ProfileStatsService({FirebaseFirestore? firestore})
      : _firestoreOverride = firestore;

  static const int xpPerLevel = 100;

  final FirebaseFirestore? _firestoreOverride;

  // Resolved lazily so the service can be constructed without Firebase
  // (e.g. in widget tests that override the load methods).
  FirebaseFirestore get _firestore =>
      _firestoreOverride ?? FirebaseFirestore.instance;

  /// Level 1-4: Beginner, 5-9: Signer, 10+: Fluent.
  LevelInfo levelForExp(int exp) {
    final level = exp ~/ xpPerLevel + 1;
    final title = level >= 10
        ? 'Fluent'
        : level >= 5
            ? 'Signer'
            : 'Beginner';
    return LevelInfo(
      level: level,
      title: title,
      xpIntoLevel: exp % xpPerLevel,
      xpForLevel: xpPerLevel,
    );
  }

  CollectionReference<Map<String, dynamic>> _attempts(String uid) =>
      _firestore.collection('users').doc(uid).collection('attempts');

  /// Stats that only depend on attempts (streaks, weekly XP, ...).
  ProfileStats _activityStats(QuerySnapshot<Map<String, dynamic>> attempts) {
    return ProfileStats.compute(
      activity: attempts.docs
          .map((doc) => ActivityRecord.fromMap(doc.data()))
          .whereType<ActivityRecord>(),
      progress: const [],
      masteryThreshold: 1,
      now: DateTime.now(),
    );
  }

  /// Live current streak, recomputed whenever an attempt is recorded.
  Stream<int> watchCurrentStreak(String uid) {
    return _attempts(uid)
        .snapshots()
        .map((snapshot) => _activityStats(snapshot).currentStreak);
  }

  Future<ProfileStats> loadStats(String uid) async {
    final user = _firestore.collection('users').doc(uid);
    final results = await Future.wait([
      user.collection('attempts').get(),
      user.collection('letterProgress').get(),
      _firestore.collection('learningConfig').doc('default').get(),
    ]);
    final attempts = results[0] as QuerySnapshot<Map<String, dynamic>>;
    final progress = results[1] as QuerySnapshot<Map<String, dynamic>>;
    final config = results[2] as DocumentSnapshot<Map<String, dynamic>>;
    final configData = config.data();
    if (configData == null) {
      throw StateError('Firestore document learningConfig/default is missing.');
    }

    return ProfileStats.compute(
      activity: attempts.docs
          .map((doc) => ActivityRecord.fromMap(doc.data()))
          .whereType<ActivityRecord>(),
      progress: progress.docs.map((doc) => LetterProgress.fromMap(doc.data())),
      masteryThreshold: LearningConfig.fromMap(configData).masteryThreshold,
      now: DateTime.now(),
    );
  }

  /// Only the current user for now: Firestore rules keep each user's data
  /// private, so other players' weekly XP cannot be read from the client.
  Future<List<LeaderboardEntry>> loadLeaderboard({
    required String uid,
    required String currentUserName,
  }) async {
    final attempts = await _attempts(uid).get();
    return [
      LeaderboardEntry(
        name: currentUserName,
        weeklyXp: _activityStats(attempts).weeklyXp,
        isCurrentUser: true,
      ),
    ];
  }
}
