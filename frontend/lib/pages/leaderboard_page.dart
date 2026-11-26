import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/profile_stats_service.dart';
import '../services/user_profile_service.dart';

// TODO: Page temporaire copier de l'ancier profile
class LeaderboardPage extends StatefulWidget {
  const LeaderboardPage({
    super.key,
    this.statsService,
    this.profileService,
  });

  final ProfileStatsService? statsService;
  final UserProfileService? profileService;

  @override
  State<LeaderboardPage> createState() => _LeaderboardPageState();
}

class _LeaderboardPageState extends State<LeaderboardPage> {
  late final _stats = widget.statsService ?? ProfileStatsService();
  late final _profiles = widget.profileService ?? UserProfileService();
  late final Future<List<LeaderboardEntry>> _entries = _load();

  Future<List<LeaderboardEntry>> _load() async {
    final user = AuthService().currentUser;
    if (user == null) throw StateError('No signed-in user was found.');
    final profile = await _profiles.loadOrCreateProfile(user);
    return _stats.loadLeaderboard(
      uid: user.uid,
      currentUserName: profile.displayName,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Leaderboard')),
      body: FutureBuilder<List<LeaderboardEntry>>(
        future: _entries,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Could not load leaderboard: ${snapshot.error}'),
              ),
            );
          }
          return SingleChildScrollView(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 620),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Card(
                    margin: EdgeInsets.zero,
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Text(
                            'Weekly leaderboard',
                            style: Theme.of(context).textTheme.titleLarge,
                          ),
                          const SizedBox(height: 16),
                          LeaderboardList(entries: snapshot.data ?? const []),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class LeaderboardList extends StatelessWidget {
  const LeaderboardList({super.key, required this.entries});

  final List<LeaderboardEntry> entries;

  static const _medalColors = [
    Color(0xFFFFC107), // gold
    Color(0xFFB0BEC5), // silver
    Color(0xFFCD7F32), // bronze
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (var i = 0; i < entries.length; i++)
          Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: entries[i].isCurrentUser
                  ? colors.primaryContainer
                  : colors.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 32,
                  child: i < 3
                      ? Icon(Icons.workspace_premium, color: _medalColors[i])
                      : Text(
                          '${i + 1}',
                          textAlign: TextAlign.center,
                          style: theme.textTheme.titleMedium,
                        ),
                ),
                const SizedBox(width: 8),
                CircleAvatar(
                  radius: 16,
                  backgroundColor: colors.secondaryContainer,
                  child: Text(
                    entries[i].name.isEmpty
                        ? '?'
                        : entries[i].name[0].toUpperCase(),
                    style: TextStyle(color: colors.onSecondaryContainer),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    entries[i].isCurrentUser
                        ? '${entries[i].name} (you)'
                        : entries[i].name,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodyLarge?.copyWith(
                      fontWeight: entries[i].isCurrentUser
                          ? FontWeight.bold
                          : FontWeight.normal,
                    ),
                  ),
                ),
                Text(
                  '${entries[i].weeklyXp} XP',
                  style: theme.textTheme.labelLarge,
                ),
              ],
            ),
          ),
        const SizedBox(height: 4),
        Text(
          'Other players will show up here once the leaderboard goes live.',
          style: theme.textTheme.bodySmall?.copyWith(
            color: colors.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}
