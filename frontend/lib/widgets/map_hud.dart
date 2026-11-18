import 'package:flutter/material.dart';
import 'package:frontend/services/auth_service.dart';
import 'package:frontend/services/user_profile_service.dart';

class MapHud extends StatefulWidget {
  const MapHud({super.key});

  @override
  State<MapHud> createState() => _MapHudState();
}

class _MapHudState extends State<MapHud> {
  late final Future<UserProfile>? _profile;

  @override
  void initState() {
    super.initState();
    final user = AuthService().currentUser;
    _profile =
        user == null ? null : UserProfileService().loadOrCreateProfile(user);
  }

  String _xpValue(AsyncSnapshot<UserProfile> snapshot) {
    if (_profile == null) return '—';
    if (snapshot.hasError) return '!';
    if (snapshot.connectionState == ConnectionState.waiting) return '…';
    final profile = snapshot.data;
    return profile == null ? '!' : '${profile.exp}';
  }

  String _xpTooltip(AsyncSnapshot<UserProfile> snapshot) {
    if (_profile == null) return 'No signed-in profile';
    if (snapshot.hasError) return 'Could not load XP: ${snapshot.error}';
    if (snapshot.data == null) return 'Profile data unavailable';
    return 'Experience points';
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.topLeft,
      child: SafeArea(
        child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
                // mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: colors.surface,
                      border: Border.all(color: colors.primary, width: 5),
                      borderRadius: BorderRadius.circular(8),
                      boxShadow: const [
                        BoxShadow(
                          color: Color(0x55000000),
                          blurRadius: 8,
                          offset: Offset(2, 4),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.auto_awesome_outlined,
                          color: colors.primary,
                        ),
                        const SizedBox(width: 10),
                        Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            FutureBuilder<UserProfile>(
                              future: _profile,
                              builder: (context, snapshot) {
                                return Tooltip(
                                  message: _xpTooltip(snapshot),
                                  child: Text(
                                    _xpValue(snapshot),
                                    style:
                                        Theme.of(context).textTheme.labelLarge,
                                  ),
                                );
                              },
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  SizedBox(width: 10),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: colors.surface,
                      border: Border.all(color: colors.primary, width: 5),
                      borderRadius: BorderRadius.circular(8),
                      boxShadow: const [
                        BoxShadow(
                          color: Color(0x55000000),
                          blurRadius: 8,
                          offset: Offset(2, 4),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.local_fire_department_outlined,
                          color: colors.primary,
                        ),
                        const SizedBox(width: 10),
                        Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '67',
                              style: Theme.of(context).textTheme.labelLarge,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ])),
      ),
    );
  }
}
