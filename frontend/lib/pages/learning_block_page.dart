import 'package:flutter/material.dart';
import 'package:frontend/models/learning_models.dart';
import 'package:frontend/services/auth_service.dart';
import 'package:frontend/services/learning_service.dart';

import 'focused_practice_session_page.dart';

class LearningBlockPage extends StatefulWidget {
  const LearningBlockPage({this.learningService, super.key});

  final LearningService? learningService;

  @override
  State<LearningBlockPage> createState() => _LearningBlockPageState();
}

class _LearningBlockPageState extends State<LearningBlockPage> {
  late final Future<LearningBlock>? _blockFuture;
  String? _error;

  @override
  void initState() {
    super.initState();
    final uid = AuthService().currentUser?.uid;
    if (uid == null) {
      _error = 'Sign in to generate a personal learning block.';
      return;
    }
    _blockFuture = (widget.learningService ?? LearningService())
        .generateLearningBlock(uid, language: 'ASL');
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) return _messagePage(_error!);

    return FutureBuilder<LearningBlock>(
      future: _blockFuture,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return _messagePage(
            'Could not generate your learning block: ${snapshot.error}',
          );
        }
        if (!snapshot.hasData) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        final block = snapshot.data!;
        if (block.letters.isEmpty) {
          return _messagePage(
            block.activeLetterCount == 0
                ? 'No active ASL letters are configured yet.'
                : 'All active ASL letters are mastered. Check back later!',
          );
        }

        return FocusedPracticeSessionPage(
          letters:
              block.letters.map((item) => item.letter).toList(growable: false),
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
