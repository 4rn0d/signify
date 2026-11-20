import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/profile_stats_service.dart';
import '../services/user_profile_service.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key, required this.email, required this.profile});

  final String? email;
  final Future<UserProfile> profile;

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfileData {
  const _ProfileData({
    required this.profile,
    required this.stats,
    required this.leaderboard,
  });

  final UserProfile profile;
  final ProfileStats stats;
  final List<LeaderboardEntry> leaderboard;
}

class _AvatarOption {
  const _AvatarOption(this.id, this.icon, this.color);

  /// Stable id stored in Firestore (users/{uid}.iconId). Never rename or
  /// reuse an id, or users who picked it will get a different icon.
  final String id;
  final IconData icon;
  final Color color;
}

// The first option is the default when the user has no saved iconId.
const _avatarOptions = <_AvatarOption>[
  _AvatarOption('person', Icons.person, Colors.indigo),
  _AvatarOption('face', Icons.face, Colors.teal),
  _AvatarOption('pets', Icons.pets, Colors.orange),
  _AvatarOption('rocket', Icons.rocket_launch, Colors.deepPurple),
  _AvatarOption('star', Icons.star, Colors.amber),
  _AvatarOption('nature', Icons.emoji_nature, Colors.green),
  _AvatarOption('gamepad', Icons.sports_esports, Colors.red),
  _AvatarOption('music', Icons.music_note, Colors.pink),
  _AvatarOption('wave', Icons.waving_hand, Colors.blue),
  _AvatarOption('fire', Icons.local_fire_department, Colors.deepOrange),
  _AvatarOption('diamond', Icons.diamond, Colors.cyan),
  _AvatarOption('heart', Icons.favorite, Colors.redAccent),
];

class _ProfilePageState extends State<ProfilePage> {
  final _service = ProfileStatsService();
  final _profiles = UserProfileService();
  late final Future<_ProfileData> _data;
  int _avatarIndex = 0;

  @override
  void initState() {
    super.initState();
    _data = _load();
  }

  /// Index of the saved avatar, or 0 (the first icon) when the profile has
  /// no iconId yet or the id is unknown.
  int _avatarIndexFor(String? iconId) {
    final index = _avatarOptions.indexWhere((option) => option.id == iconId);
    return index == -1 ? 0 : index;
  }

  /// Fetches the saved icon id from Firestore each time the page opens.
  /// Falls back to the id already on [profile] if the fetch fails, and the
  /// caller then falls back to the first icon when that is null too.
  Future<String?> _fetchIconId(UserProfile profile) async {
    final user = AuthService().currentUser;
    if (user == null) return profile.iconId;
    try {
      return await _profiles.fetchIconId(user);
    } catch (_) {
      return profile.iconId;
    }
  }

  Future<_ProfileData> _load() async {
    final profile = await widget.profile;
    _avatarIndex = _avatarIndexFor(await _fetchIconId(profile));
    final stats = await _service.loadStats();
    final leaderboard = await _service.loadLeaderboard(
      currentUserName: profile.displayName,
    );
    return _ProfileData(
      profile: profile,
      stats: stats,
      leaderboard: leaderboard,
    );
  }

  Future<void> _pickAvatar() async {
    final picked = await showDialog<int>(
      context: context,
      builder: (context) => _AvatarPickerDialog(selected: _avatarIndex),
    );
    if (picked == null || picked == _avatarIndex || !mounted) return;

    final previous = _avatarIndex;
    setState(() => _avatarIndex = picked);
    try {
      final user = AuthService().currentUser;
      if (user == null) throw StateError('No signed-in user.');
      await _profiles.saveIconId(user, _avatarOptions[picked].id);
    } catch (error) {
      if (!mounted) return;
      setState(() => _avatarIndex = previous);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save avatar: $error')),
      );
    }
  }

  void _openSettings() {
    // TODO: navigate to the settings page once it exists.
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Settings coming soon.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Settings',
            onPressed: _openSettings,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: FutureBuilder<_ProfileData>(
        future: _data,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Could not load profile: ${snapshot.error}'),
              ),
            );
          }
          final data = snapshot.data;
          if (data == null) {
            return const Center(child: Text('Profile not found.'));
          }

          final level = _service.levelForExp(data.profile.exp);

          return SingleChildScrollView(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1100),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final wide = constraints.maxWidth >= 800;

                      final header = _HeaderCard(
                        name: data.profile.displayName,
                        email: widget.email,
                        level: level,
                        avatar: _avatarOptions[_avatarIndex],
                        onEditAvatar: _pickAvatar,
                      );
                      final stats = _SectionCard(
                        title: 'Your stats',
                        child: _StatsWrap(
                          exp: data.profile.exp,
                          level: level,
                          stats: data.stats,
                        ),
                      );
                      final calendar = _SectionCard(
                        title: 'Streak calendar',
                        child: _StreakCalendar(
                          month: data.stats.calendarMonth,
                          activeDays: data.stats.activeDays,
                        ),
                      );
                      final leaderboard = _SectionCard(
                        title: 'Weekly leaderboard',
                        child: _Leaderboard(entries: data.leaderboard),
                      );

                      if (!wide) {
                        return Column(
                          children: [
                            header,
                            const SizedBox(height: 16),
                            stats,
                            const SizedBox(height: 16),
                            calendar,
                            const SizedBox(height: 16),
                            leaderboard,
                          ],
                        );
                      }

                      return Column(
                        children: [
                          header,
                          const SizedBox(height: 24),
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                flex: 3,
                                child: Column(
                                  children: [
                                    stats,
                                    const SizedBox(height: 24),
                                    calendar,
                                  ],
                                ),
                              ),
                              const SizedBox(width: 24),
                              Expanded(flex: 2, child: leaderboard),
                            ],
                          ),
                        ],
                      );
                    },
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

// ---------------------------------------------------------------------------
// Header: avatar, name, level + XP progress bar
// ---------------------------------------------------------------------------

class _HeaderCard extends StatelessWidget {
  const _HeaderCard({
    required this.name,
    required this.email,
    required this.level,
    required this.avatar,
    required this.onEditAvatar,
  });

  final String name;
  final String? email;
  final LevelInfo level;
  final _AvatarOption avatar;
  final VoidCallback onEditAvatar;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Row(
          children: [
            Tooltip(
              message: 'Change avatar',
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: onEditAvatar,
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    CircleAvatar(
                      radius: 56,
                      backgroundColor: avatar.color,
                      child: Icon(avatar.icon, size: 60, color: Colors.white),
                    ),
                    Positioned(
                      right: -2,
                      bottom: -2,
                      child: CircleAvatar(
                        radius: 18,
                        backgroundColor: colors.primary,
                        child: Icon(
                          Icons.edit,
                          size: 18,
                          color: colors.onPrimary,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(width: 32),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: theme.textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    email ?? 'No email available',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Chip(
                        avatar: Icon(
                          Icons.military_tech_outlined,
                          color: colors.primary,
                        ),
                        label: Text('Level ${level.level} · ${level.title}'),
                      ),
                      const Spacer(),
                      Text(
                        '${level.xpIntoLevel} / ${level.xpForLevel} XP',
                        style: theme.textTheme.labelLarge,
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: level.progress,
                      minHeight: 14,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    '${level.xpToNext} XP to level ${level.level + 1}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Avatar picker
// ---------------------------------------------------------------------------

class _AvatarPickerDialog extends StatelessWidget {
  const _AvatarPickerDialog({required this.selected});

  final int selected;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return AlertDialog(
      title: const Text('Choose your avatar'),
      content: SizedBox(
        width: 420,
        child: Wrap(
          spacing: 16,
          runSpacing: 16,
          children: [
            for (var i = 0; i < _avatarOptions.length; i++)
              InkWell(
                customBorder: const CircleBorder(),
                onTap: () => Navigator.of(context).pop(i),
                child: Container(
                  padding: const EdgeInsets.all(3),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: i == selected
                          ? colors.primary
                          : Colors.transparent,
                      width: 3,
                    ),
                  ),
                  child: CircleAvatar(
                    radius: 28,
                    backgroundColor: _avatarOptions[i].color,
                    child: Icon(
                      _avatarOptions[i].icon,
                      color: Colors.white,
                      size: 30,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Shared section card
// ---------------------------------------------------------------------------

class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.title, required this.child});

  final String title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: Card(
        margin: EdgeInsets.zero,
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              child,
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Stats tiles
// ---------------------------------------------------------------------------

String _formatMinutes(int minutes) {
  final h = minutes ~/ 60;
  final m = minutes % 60;
  return h == 0 ? '${m}m' : '${h}h ${m}m';
}

class _StatsWrap extends StatelessWidget {
  const _StatsWrap({
    required this.exp,
    required this.level,
    required this.stats,
  });

  final int exp;
  final LevelInfo level;
  final ProfileStats stats;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: [
        _StatTile(
          icon: Icons.auto_awesome_outlined,
          label: 'Total XP',
          value: '$exp',
        ),
        _StatTile(
          icon: Icons.military_tech_outlined,
          label: 'Current level',
          value: '${level.level}',
        ),
        _StatTile(
          icon: Icons.local_fire_department_outlined,
          label: 'Daily streak',
          value: '${stats.currentStreak} days',
        ),
        _StatTile(
          icon: Icons.emoji_events_outlined,
          label: 'Longest streak',
          value: '${stats.longestStreak} days',
        ),
        _StatTile(
          icon: Icons.timer_outlined,
          label: 'Time learning',
          value: _formatMinutes(stats.minutesLearned),
        ),
        _StatTile(
          icon: Icons.sign_language_outlined,
          label: 'Signs learned',
          value: '${stats.signsLearned}',
        ),
        _StatTile(
          icon: Icons.check_circle_outline,
          label: 'Lessons completed',
          value: '${stats.lessonsCompleted}',
        ),
        _StatTile(
          icon: Icons.track_changes_outlined,
          label: 'Quiz accuracy',
          value: '${(stats.quizAccuracy * 100).round()}%',
        ),
      ],
    );
  }
}

class _StatTile extends StatelessWidget {
  const _StatTile({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return SizedBox(
      width: 160,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: colors.primary),
            const SizedBox(height: 10),
            Text(
              value,
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: theme.textTheme.bodySmall?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Streak calendar
// ---------------------------------------------------------------------------

class _StreakCalendar extends StatelessWidget {
  const _StreakCalendar({required this.month, required this.activeDays});

  final DateTime month;
  final Set<int> activeDays;

  static const _monthNames = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ];
  static const _weekdays = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    final daysInMonth = DateTime(month.year, month.month + 1, 0).day;
    final leadingBlanks = DateTime(month.year, month.month, 1).weekday - 1;
    final now = DateTime.now();
    final isCurrentMonth = now.year == month.year && now.month == month.month;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '${_monthNames[month.month - 1]} ${month.year}',
          style: theme.textTheme.titleMedium,
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            for (final d in _weekdays)
              Expanded(
                child: Center(
                  child: Text(
                    d,
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 8),
        GridView.count(
          crossAxisCount: 7,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          mainAxisSpacing: 6,
          crossAxisSpacing: 6,
          childAspectRatio: 1.5,
          children: [
            for (var i = 0; i < leadingBlanks; i++) const SizedBox.shrink(),
            for (var day = 1; day <= daysInMonth; day++)
              _DayCell(
                day: day,
                active: activeDays.contains(day),
                isToday: isCurrentMonth && now.day == day,
              ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Icon(
              Icons.local_fire_department,
              size: 16,
              color: colors.primary,
            ),
            const SizedBox(width: 6),
            Text(
              'Practiced ${activeDays.length} days this month',
              style: theme.textTheme.bodySmall?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _DayCell extends StatelessWidget {
  const _DayCell({
    required this.day,
    required this.active,
    required this.isToday,
  });

  final int day;
  final bool active;
  final bool isToday;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Container(
      decoration: BoxDecoration(
        color: active ? colors.primary : colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
        border: isToday ? Border.all(color: colors.tertiary, width: 2) : null,
      ),
      alignment: Alignment.center,
      child: Text(
        '$day',
        style: TextStyle(
          color: active ? colors.onPrimary : colors.onSurfaceVariant,
          fontWeight: active ? FontWeight.bold : FontWeight.normal,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Weekly leaderboard
// ---------------------------------------------------------------------------

class _Leaderboard extends StatelessWidget {
  const _Leaderboard({required this.entries});

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
      ],
    );
  }
}