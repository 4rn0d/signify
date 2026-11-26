import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:frontend/models/learning_models.dart';

class LearningService {
  LearningService({FirebaseFirestore? firestore})
      : _firestore = firestore ?? FirebaseFirestore.instance;

  final FirebaseFirestore _firestore;

  DocumentReference<Map<String, dynamic>> get _configDocument =>
      _firestore.collection('learningConfig').doc('default');

  CollectionReference<Map<String, dynamic>> get _letters =>
      _firestore.collection('letters');

  CollectionReference<Map<String, dynamic>> get _words =>
      _firestore.collection('words');

  CollectionReference<Map<String, dynamic>> _userCollection(String uid) {
    if (uid.trim().isEmpty) {
      throw ArgumentError.value(uid, 'uid', 'A signed-in user is required.');
    }
    return _firestore.collection('users').doc(uid).collection('letterProgress');
  }

  Future<LearningConfig> loadConfig() async {
    final snapshot = await _configDocument.get();
    final data = snapshot.data();
    if (data == null) {
      throw StateError('Firestore document learningConfig/default is missing.');
    }
    return LearningConfig.fromMap(data);
  }

  Future<LearningBlock> generateLearningBlock(
    String uid, {
    String? language,
  }) async {
    final config = await loadConfig();
    final letterSnapshot =
        await _letters.where('isActive', isEqualTo: true).get();
    final activeLetters = letterSnapshot.docs
        .map((doc) => LearningLetter.fromMap(doc.id, doc.data()))
        .where(
            (letter) => language == null || letter.matchesLanguage(language));
    final progress = await _loadProgress(uid);

    return LearningBlock.generate(
      activeLetters: activeLetters,
      progressByLetterId: progress,
      config: config,
    );
  }

  Future<List<LearningWord>> eligibleWords(
    String uid, {
    String? language,
  }) async {
    final config = await loadConfig();
    final wordSnapshot = await _words.where('isActive', isEqualTo: true).get();
    final progress = await _loadProgress(uid);

    return wordSnapshot.docs
        .map((doc) => LearningWord.fromMap(doc.id, doc.data()))
        .where((word) => language == null || word.language == language)
        .where(
          (word) => word.isEligible(
            progressByLetterId: progress,
            spellingEligibilityThreshold: config.spellingEligibilityThreshold,
          ),
        )
        .toList(growable: false);
  }

  Future<void> recordLetterAttempt({
    required String uid,
    required String exerciseType,
    required String letterId,
    required bool isCorrect,
  }) {
    if (exerciseType.trim().isEmpty) {
      throw ArgumentError.value(
        exerciseType,
        'exerciseType',
        'An exercise type is required.',
      );
    }
    if (letterId.trim().isEmpty) {
      throw ArgumentError.value(
        letterId,
        'letterId',
        'A letter document ID is required.',
      );
    }
    return _recordAttempt(
      uid: uid,
      attempt: {
        'exerciseType': exerciseType,
        'letterIds': [letterId],
        'isCorrect': isCorrect,
      },
      results: [LetterAttemptResult(letterId: letterId, isCorrect: isCorrect)],
    );
  }

  Future<void> recordSpellingAttempt({
    required String uid,
    required String wordId,
    required List<LetterAttemptResult> letterResults,
  }) {
    if (wordId.trim().isEmpty) {
      throw ArgumentError.value(
          wordId, 'wordId', 'A word document ID is required.');
    }
    if (letterResults.isEmpty) {
      throw ArgumentError.value(
        letterResults,
        'letterResults',
        'At least one letter result is required.',
      );
    }
    if (letterResults.any((result) => result.letterId.trim().isEmpty)) {
      throw ArgumentError.value(
        letterResults,
        'letterResults',
        'Every result must include a letter document ID.',
      );
    }
    return _recordAttempt(
      uid: uid,
      attempt: {
        'exerciseType': 'spelling',
        'wordId': wordId,
        'letterResults': letterResults.map((result) => result.toMap()).toList(),
      },
      results: letterResults,
    );
  }

  Future<void> recordLetterMatchingAttempt({
    required String uid,
    required List<LetterAttemptResult> letterResults,
  }) {
    if (letterResults.isEmpty) {
      throw ArgumentError.value(
        letterResults,
        'letterResults',
        'At least one letter result is required.',
      );
    }
    if (letterResults.any((result) => result.letterId.trim().isEmpty)) {
      throw ArgumentError.value(
        letterResults,
        'letterResults',
        'Every result must include a letter document ID.',
      );
    }
    return _recordAttempt(
      uid: uid,
      attempt: {
        'exerciseType': 'focused_practice_multi_letter_matching',
        'letterIds': letterResults.map((result) => result.letterId).toList(),
        'letterResults': letterResults.map((result) => result.toMap()).toList(),
      },
      results: letterResults,
    );
  }

  Future<void> _recordAttempt({
    required String uid,
    required Map<String, Object> attempt,
    required List<LetterAttemptResult> results,
  }) async {
    final progressCollection = _userCollection(uid);
    final attemptCollection =
        _firestore.collection('users').doc(uid).collection('attempts');
    final attemptDocument = attemptCollection.doc();
    final configDocument = _configDocument;
    final progressDocuments = {
      for (final result in results)
        result.letterId: progressCollection.doc(result.letterId),
    };

    await _firestore.runTransaction((transaction) async {
      final configSnapshot = await transaction.get(configDocument);
      final configData = configSnapshot.data();
      if (configData == null) {
        throw StateError(
            'Firestore document learningConfig/default is missing.');
      }
      final config = LearningConfig.fromMap(configData);

      final snapshots = <String, DocumentSnapshot<Map<String, dynamic>>>{};
      for (final entry in progressDocuments.entries) {
        snapshots[entry.key] = await transaction.get(entry.value);
      }

      final resultsByLetter = <String, List<bool>>{};
      for (final result in results) {
        resultsByLetter
            .putIfAbsent(result.letterId, () => [])
            .add(result.isCorrect);
      }

      for (final entry in progressDocuments.entries) {
        final snapshot = snapshots[entry.key]!;
        final current = snapshot.data() == null
            ? const LetterProgress.empty()
            : LetterProgress.fromMap(snapshot.data()!);
        final updated = current.applyResults(
          resultsByLetter[entry.key]!,
          masteryThreshold: config.masteryThreshold,
        );
        transaction.set(entry.value, {
          'masteryScore': updated.masteryScore,
          'totalAttempts': updated.totalAttempts,
          'correctAttempts': updated.correctAttempts,
          'incorrectAttempts': updated.incorrectAttempts,
          'lastPracticedAt': FieldValue.serverTimestamp(),
          'updatedAt': FieldValue.serverTimestamp(),
        });
      }

      transaction.set(attemptDocument, {
        ...attempt,
        'createdAt': FieldValue.serverTimestamp(),
      });
    });
  }

  Future<Map<String, LetterProgress>> _loadProgress(String uid) async {
    final snapshot = await _userCollection(uid).get();
    return {
      for (final doc in snapshot.docs)
        doc.id: LetterProgress.fromMap(doc.data()),
    };
  }
}
