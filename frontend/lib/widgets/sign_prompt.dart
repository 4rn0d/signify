import 'package:flutter/material.dart';

class SignPrompt extends StatelessWidget {
  const SignPrompt({
    required this.letter,
    required this.letterColor,
    required this.borderColor,
    this.prompt = 'Can you make this sign?',
    super.key,
  });

  final String letter;
  final Color letterColor;
  final Color borderColor;
  final String prompt;

  @override
  Widget build(BuildContext context) {
    final surfaceColor = Theme.of(context).colorScheme.surface;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
          decoration: BoxDecoration(
            color: surfaceColor,
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: borderColor),
            boxShadow: const [
              BoxShadow(
                color: Color(0x12000000),
                blurRadius: 14,
                offset: Offset(0, 5),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                letter,
                style: Theme.of(context)
                    .textTheme
                    .headlineMedium
                    ?.copyWith(color: letterColor, fontWeight: FontWeight.w900),
              ),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  prompt,
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontWeight: FontWeight.w600),
                ),
              ),
              const SizedBox(width: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Image.asset(
                  'assets/asl-signs/$letter.png',
                  height: 64,
                  width: 52,
                  fit: BoxFit.contain,
                ),
              ),
            ],
          ),
        ),
        Transform.rotate(
          angle: 0.785398,
          child: Container(
            width: 16,
            height: 16,
            margin: const EdgeInsets.only(top: 2),
            decoration: BoxDecoration(
              color: surfaceColor,
              border: Border(
                right: BorderSide(color: borderColor),
                bottom: BorderSide(color: borderColor),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
