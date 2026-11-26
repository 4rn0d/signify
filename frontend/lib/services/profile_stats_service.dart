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

class ProfileStats {
  const ProfileStats({
    required this.currentStreak,
    required this.longestStreak,
    required this.minutesLearned,
    required this.signsLearned,
    required this.lessonsCompleted,
    required this.quizAccuracy,
    required this.calendarMonth,
    required this.activeDays,
  });

  final int currentStreak;
  final int longestStreak;
  final int minutesLearned;
  final int signsLearned;
  final int lessonsCompleted;

  /// 0.0 - 1.0
  final double quizAccuracy;

  /// Month shown in the streak calendar (day is always 1).
  final DateTime calendarMonth;

  /// Days of [calendarMonth] (1-31) on which the user practiced.
  final Set<int> activeDays;
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
  static const int xpPerLevel = 100;

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

  Future<ProfileStats> loadStats() async {
    await Future<void>.delayed(const Duration(milliseconds: 300));
    final now = DateTime.now();
    return ProfileStats(
      currentStreak: 12,
      longestStreak: 21,
      minutesLearned: 745,
      signsLearned: 128,
      lessonsCompleted: 34,
      quizAccuracy: 0.87,
      calendarMonth: DateTime(now.year, now.month),
      activeDays: {
        for (var d = 1; d <= now.day; d++)
          if (d % 7 != 3) d,
      },
    );
  }

  Future<List<LeaderboardEntry>> loadLeaderboard({
    required String currentUserName,
  }) async {
    await Future<void>.delayed(const Duration(milliseconds: 300));
    final entries = <LeaderboardEntry>[
      const LeaderboardEntry(name: 'Emma', weeklyXp: 520),
      const LeaderboardEntry(name: 'Lucas', weeklyXp: 465),
      LeaderboardEntry(
        name: currentUserName,
        weeklyXp: 340,
        isCurrentUser: true,
      ),
      const LeaderboardEntry(name: 'Chloé', weeklyXp: 310),
      const LeaderboardEntry(name: 'Noah', weeklyXp: 275),
      const LeaderboardEntry(name: 'Léa', weeklyXp: 190),
      const LeaderboardEntry(name: 'Thomas', weeklyXp: 120),
    ];
    entries.sort((a, b) => b.weeklyXp.compareTo(a.weeklyXp));
    return entries;
  }
}