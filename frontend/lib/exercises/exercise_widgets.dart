import 'package:flutter/material.dart';

const practiceCorrectColor = Color(0xFF2E7D32);
const practiceIncorrectColor = Color(0xFFC62828);

class ExerciseSignImage extends StatelessWidget {
  const ExerciseSignImage({
    required this.character,
    required this.width,
    required this.height,
    this.semanticLabel,
    super.key,
  });

  final String character;
  final double width;
  final double height;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) => SizedBox(
        width: width,
        height: height,
        child: Image.asset(
          'assets/asl-signs/$character.png',
          fit: BoxFit.contain,
          filterQuality: FilterQuality.none,
          semanticLabel: semanticLabel,
        ),
      );
}
