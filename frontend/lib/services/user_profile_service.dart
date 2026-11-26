import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserProfile {
  const UserProfile({
    required this.displayName,
    required this.exp,
    this.iconId,
  });

  final String displayName;
  final int exp;

  /// Id of the avatar icon the user picked. Null until they pick one, in
  /// which case the UI falls back to the first icon.
  final String? iconId;

  factory UserProfile.fromMap(Map<String, dynamic> data) {
    final displayName = data['displayName'];
    final exp = data['exp'];
    final iconId = data['iconId'];
    if (displayName is! String || exp is! num) {
      throw const FormatException('User profile has invalid fields.');
    }

    return UserProfile(
      displayName: displayName,
      exp: exp.toInt(),
      iconId: iconId is String ? iconId : null,
    );
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

  Stream<UserProfile> watchProfile(User user) {
    return _users.doc(user.uid).snapshots().map((snapshot) {
      final data = snapshot.data();
      if (data == null) {
        throw StateError('User profile does not exist.');
      }
      return UserProfile.fromMap(data);
    });
  }

  Future<void> saveIconId(User user, String iconId) {
    return _users.doc(user.uid).set({
      'iconId': iconId,
      'updatedAt': FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));
  }

  Future<String?> fetchIconId(User user) async {
    final snapshot = await _users.doc(user.uid).get();
    final iconId = snapshot.data()?['iconId'];
    return iconId is String ? iconId : null;
  }

  String _defaultDisplayName(String? email) {
    final emailName = email?.split('@').first.trim();
    return emailName == null || emailName.isEmpty ? 'Player' : emailName;
  }
}
