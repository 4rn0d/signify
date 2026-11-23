import 'package:flutter/material.dart';

const practiceCorrectColor = Color(0xFF2E7D32);
const practiceCorrectBackgroundColor = Color(0xFFE8F5E9);
const practiceIncorrectColor = Color(0xFFC62828);
const practiceIncorrectBackgroundColor = Color(0xFFFFEBEE);

class PracticePage extends StatelessWidget {
  const PracticePage({
    required this.title,
    required this.questionNumber,
    required this.questionCount,
    required this.score,
    required this.questionContent,
    required this.answered,
    required this.isCorrect,
    required this.feedback,
    required this.saving,
    required this.canConfirm,
    required this.onConfirm,
    required this.continueLabel,
    required this.onContinue,
    super.key,
  });

  final String title;
  final int questionNumber;
  final int questionCount;
  final int score;
  final Widget questionContent;
  final bool answered;
  final bool isCorrect;
  final String feedback;
  final bool saving;
  final bool canConfirm;
  final VoidCallback onConfirm;
  final String continueLabel;
  final VoidCallback onContinue;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 930),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _PracticeProgress(
                    questionNumber: questionNumber,
                    questionCount: questionCount,
                    score: score,
                  ),
                  const SizedBox(height: 20),
                  Expanded(child: questionContent),
                  const SizedBox(height: 12),
                  _PracticeConfirmation(
                    answered: answered,
                    isCorrect: isCorrect,
                    feedback: feedback,
                    saving: saving,
                    canConfirm: canConfirm,
                    onConfirm: onConfirm,
                    continueLabel: continueLabel,
                    onContinue: onContinue,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _PracticeProgress extends StatelessWidget {
  const _PracticeProgress({
    required this.questionNumber,
    required this.questionCount,
    required this.score,
  });

  final int questionNumber;
  final int questionCount;
  final int score;

  @override
  Widget build(BuildContext context) {
    return Column(
      key: const ValueKey('lesson-progress'),
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Text(
              'QUESTION $questionNumber OF $questionCount',
              style: Theme.of(context).textTheme.labelLarge,
            ),
            const Spacer(),
            Text('Score $score'),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: questionNumber / questionCount,
            minHeight: 8,
          ),
        ),
      ],
    );
  }
}

class _PracticeConfirmation extends StatelessWidget {
  const _PracticeConfirmation({
    required this.answered,
    required this.isCorrect,
    required this.feedback,
    required this.saving,
    required this.canConfirm,
    required this.onConfirm,
    required this.continueLabel,
    required this.onContinue,
  });

  final bool answered;
  final bool isCorrect;
  final String feedback;
  final bool saving;
  final bool canConfirm;
  final VoidCallback onConfirm;
  final String continueLabel;
  final VoidCallback onContinue;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      key: const ValueKey('confirmation-zone'),
      height: 116,
      child: Column(
        children: [
          const Divider(height: 1),
          Expanded(
            child: AnimatedContainer(
              key: const ValueKey('confirmation-feedback-background'),
              duration: const Duration(milliseconds: 180),
              width: double.infinity,
              margin: const EdgeInsets.symmetric(vertical: 3),
              padding: const EdgeInsets.symmetric(vertical: 3),
              decoration: BoxDecoration(
                color: answered
                    ? isCorrect
                        ? practiceCorrectBackgroundColor
                        : practiceIncorrectBackgroundColor
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 200),
                      child: answered
                          ? _AnswerFeedback(
                              key: const ValueKey('answer-feedback'),
                              message: feedback,
                              isCorrect: isCorrect,
                            )
                          : const SizedBox.expand(
                              key: ValueKey('confirmation-feedback-empty'),
                            ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  SizedBox(
                    key: const ValueKey('practice-action-button'),
                    width: 180,
                    height: 48,
                    child: FilledButton(
                      style: answered
                          ? FilledButton.styleFrom(
                              backgroundColor: isCorrect
                                  ? practiceCorrectColor
                                  : practiceIncorrectColor,
                              foregroundColor: Colors.white,
                            )
                          : null,
                      onPressed: answered
                          ? saving
                              ? null
                              : onContinue
                          : canConfirm
                              ? onConfirm
                              : null,
                      child: answered
                          ? saving
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : Text(continueLabel)
                          : const Text('Check answer'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnswerFeedback extends StatelessWidget {
  const _AnswerFeedback({
    super.key,
    required this.message,
    required this.isCorrect,
  });

  final String message;
  final bool isCorrect;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Icon(
          isCorrect ? Icons.check_circle_rounded : Icons.lightbulb_rounded,
          color: isCorrect ? practiceCorrectColor : practiceIncorrectColor,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            key: const ValueKey('answer-feedback-text'),
            message,
            style: TextStyle(
              color: isCorrect ? practiceCorrectColor : practiceIncorrectColor,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }
}
