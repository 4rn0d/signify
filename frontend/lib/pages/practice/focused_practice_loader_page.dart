import 'package:flutter/material.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/services/auth_service.dart';
import 'package:frontend/services/learning_service.dart';

import 'focused_practice_page.dart';

class FocusedPracticeLoaderPage extends StatefulWidget {
  const FocusedPracticeLoaderPage({this.learningService, super.key});

  final LearningService? learningService;

  @override
  State<FocusedPracticeLoaderPage> createState() =>
      _FocusedPracticeLoaderPageState();
}

class _FocusedPracticeLoaderPageState extends State<FocusedPracticeLoaderPage> {
  late final Future<_FocusedPracticeData>? _dataFuture;
  String? _error;

  @override
  void initState() {
    super.initState();
    final uid = AuthService().currentUser?.uid;
    if (uid == null) {
      _error = 'Sign in to generate a personal learning block.';
      _dataFuture = null;
      return;
    }
    final service = widget.learningService ?? LearningService();
    _dataFuture = _loadData(service, uid);
  }

  Future<_FocusedPracticeData> _loadData(
    LearningService service,
    String uid,
  ) async {
    final blockFuture = service.generateLearningBlock(uid, language: 'ASL');
    final wordsFuture = service.eligibleWords(uid, language: 'ASL');
    return (
      block: await blockFuture,
      words: await wordsFuture,
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) return _messagePage(_error!);

    return FutureBuilder<_FocusedPracticeData>(
      future: _dataFuture,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return _messagePage(
            'Could not load focused practice: ${snapshot.error}',
          );
        }
        if (!snapshot.hasData) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        final data = snapshot.data!;
        final block = data.block;
        if (block.letters.isEmpty) {
          return _messagePage(
            block.activeLetterCount == 0
                ? 'No active ASL letters are configured yet.'
                : 'All active ASL letters are mastered. Check back later!',
          );
        }

        return FocusedPracticePage(
          letters:
              block.letters.map((item) => item.letter).toList(growable: false),
          words: data.words,
          learningService: widget.learningService,
        );
      },
    );
  }

  Widget _messagePage(String message) => Scaffold(
        appBar: AppBar(title: const Text('Focused practice')),
        body: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: _Message(message: message),
            ),
          ),
        ),
      );
}

typedef _FocusedPracticeData = ({
  LearningBlock block,
  List<LearningWord> words,
});

class _Message extends StatelessWidget {
  const _Message({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) => Center(
        child: Text(
          message,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
      );
}
