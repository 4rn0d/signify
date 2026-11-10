import 'package:flutter/material.dart';

import '../services/user_profile_service.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key, required this.email, required this.profile});

  final String? email;
  final Future<UserProfile> profile;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: FutureBuilder<UserProfile>(
        future: profile,
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

          final userProfile = snapshot.data;
          if (userProfile == null) {
            return const Center(child: Text('Profile not found.'));
          }

          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircleAvatar(
                  radius: 48,
                  backgroundColor: Theme.of(context)
                      .colorScheme
                      .primaryContainer,
                  child: Icon(
                    Icons.person,
                    size: 56,
                    color: Theme.of(context).colorScheme.onPrimaryContainer,
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  userProfile.displayName,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text(email ?? 'No email available'),
                const SizedBox(height: 16),
                Text('${userProfile.exp} EXP'),
              ],
            ),
          );
        },
      ),
    );
  }
}
