import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserProfile {
  const UserProfile({required this.displayName, required this.exp});

  final String displayName;
  final int exp;

  factory UserProfile.fromMap(Map<String, dynamic> data) {
    final displayName = data['displayName'];
    final exp = data['exp'];
    if (displayName is! String || exp is! num) {
      throw const FormatException('User profile has invalid fields.');
    }

    return UserProfile(displayName: displayName, exp: exp.toInt());
  }
}

class UserProfileService {
  UserProfileService({FirebaseFirestore? firestore})
    : _firestore = firestore ?? FirebaseFirestore.instance;

  final FirebaseFirestore _firestore;

  CollectionReference<Map<String, dynamic>> get _users =>
      _firestore.collection('users');

  Future<void> ensureProfile(User user) async {
    final profile = _users.doc(user.uid);
    await _firestore.runTransaction((transaction) async {
      final snapshot = await transaction.get(profile);
      if (snapshot.exists) return;

      final displayName = user.displayName?.trim();
      transaction.set(profile, {
        'displayName': displayName != null && displayName.isNotEmpty
            ? displayName
            : _defaultDisplayName(user.email),
        'exp': 0,
        'createdAt': FieldValue.serverTimestamp(),
        'updatedAt': FieldValue.serverTimestamp(),
      });
    });
  }

  Future<UserProfile> loadOrCreateProfile(User user) async {
    await ensureProfile(user);
    final snapshot = await _users.doc(user.uid).get();
    final data = snapshot.data();
    if (data == null) {
      throw StateError('User profile was not created.');
    }

    return UserProfile.fromMap(data);
  }

  String _defaultDisplayName(String? email) {
    final emailName = email?.split('@').first.trim();
    return emailName == null || emailName.isEmpty ? 'Player' : emailName;
  }
}
