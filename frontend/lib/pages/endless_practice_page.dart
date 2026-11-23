import 'dart:async';

import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/capture_cleanup.dart';
import '../services/learning_service.dart';
import '../services/letter_recognizer.dart';
import '../widgets/camera_preview_panel.dart';
import '../widgets/gragas_avatar.dart';
import '../widgets/sign_prompt.dart';

class EndlessPracticePage extends StatefulWidget {
  const EndlessPracticePage({
    this.letters,
    this.practiceTitle = 'ENDLESS PRACTICE',
    super.key,
  }) : assert(letters == null || letters.length > 0);

  final List<String>? letters;
  final String practiceTitle;

  @override
  State<EndlessPracticePage> createState() => _EndlessPracticePageState();
}

class _EndlessPracticePageState extends State<EndlessPracticePage>
    with WidgetsBindingObserver {
  final _gragasAvatarKey = GlobalKey();

  CameraController? _cameraController;
  CameraMacOSController? _macOSCameraController;
  bool _cameraEnabled = false;
  bool _cameraLoading = false;
  int _currentLetterIndex = 0;
  late final List<String> _letters;
  GragasAnimation _gragasAnimation = GragasAnimation.idle;
  int _gragasAnimationTrigger = 0;
  String? _cameraError;

  final _letterRecognizer = LetterRecognizer();
  Timer? _recognitionTimer;
  bool _recognizing = false;
  int _matchStreak = 0;
  String? _recognitionStatus;

  // A letter is validated once the model sees it in this many frames in a
  // row, at this confidence or more: a single frame would pass on a flicker.
  // 80% is the threshold of the Python letter demo.
  static const _framesToValidate = 3;
  static const _minConfidence = 0.8;
  static const _recognitionInterval = Duration(milliseconds: 400);

  String get _currentLetter => _letters[_currentLetterIndex];

  void _onSignChecked(bool signCompleted) {
    unawaited(_recordLetterAttempt(signCompleted));
    if (!signCompleted) {
      setState(() {
        _matchStreak = 0;
        _recognitionStatus = 'Marked for more practice.';
      });
      return;
    }

    setState(() {
      _matchStreak = 0;
      _gragasAnimation = GragasAnimation.drinkBarrel;
      _gragasAnimationTrigger++;
      _currentLetterIndex = (_currentLetterIndex + 1) % _letters.length;
    });
  }

  Future<void> _recordLetterAttempt(bool isCorrect) async {
    final uid = AuthService().currentUser?.uid;
    if (uid == null) return;
    try {
      await LearningService().recordLetterAttempt(
        uid: uid,
        exerciseType: 'recognition',
        letterId: _currentLetter,
        isCorrect: isCorrect,
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save practice progress: $error')),
      );
    }
  }

  @override
  void initState() {
    super.initState();
    _letters = List.unmodifiable(
      widget.letters ??
          List.generate(26, (index) => String.fromCharCode(65 + index)),
    );
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _recognitionTimer?.cancel();
    _letterRecognizer.close();
    _cameraController?.dispose();
    unawaited(_macOSCameraController?.destroy());
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      unawaited(_disableCamera());
    }
  }

  Future<void> _toggleCamera() async {
    if (_cameraLoading) return;

    if (_cameraEnabled) {
      await _disableCamera();
      return;
    }

    setState(() {
      _cameraLoading = true;
      _cameraError = null;
    });

    CameraController? controller;
    try {
      if (defaultTargetPlatform == TargetPlatform.macOS) {
        if (!mounted) return;
        setState(() {
          _cameraEnabled = true;
          _cameraLoading = false;
        });
        _startRecognition();
      } else if (!kIsWeb &&
          defaultTargetPlatform != TargetPlatform.android &&
          defaultTargetPlatform != TargetPlatform.iOS &&
          defaultTargetPlatform != TargetPlatform.windows) {
        setState(() {
          _cameraLoading = false;
          _cameraError = 'Camera preview is not supported on this platform.';
        });
      } else {
        final cameras = await availableCameras();
        if (cameras.isEmpty) {
          throw CameraException(
            'NoCameraAvailable',
            'No camera was found on this device.',
          );
        }

        final camera = cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.front,
          orElse: () => cameras.first,
        );
        controller = CameraController(
          camera,
          ResolutionPreset.medium,
          enableAudio: false,
        );
        await controller.initialize();

        if (!mounted) {
          await controller.dispose();
          return;
        }

        setState(() {
          _cameraController = controller;
          _cameraEnabled = true;
          _cameraLoading = false;
        });
        _startRecognition();
      }
    } on CameraException catch (error) {
      await controller?.dispose();
      if (!mounted) return;
      setState(() {
        _cameraLoading = false;
        _cameraEnabled = false;
        _cameraController = null;
        _cameraError = _cameraErrorMessage(error);
      });
    } catch (error) {
      await controller?.dispose();
      if (!mounted) return;
      setState(() {
        _cameraLoading = false;
        _cameraEnabled = false;
        _cameraController = null;
        _cameraError = 'Could not open the camera: $error';
      });
    }
  }

  Future<void> _disableCamera() async {
    final cameraController = _cameraController;
    final macOSCameraController = _macOSCameraController;
    _recognitionTimer?.cancel();
    _recognitionTimer = null;
    if (mounted) {
      setState(() {
        _cameraEnabled = false;
        _cameraLoading = false;
        _cameraController = null;
        _macOSCameraController = null;
        _cameraError = null;
        _matchStreak = 0;
        _recognitionStatus = null;
      });
    }
    await cameraController?.dispose();
    await macOSCameraController?.destroy();
  }

  void _startRecognition() {
    _recognitionTimer?.cancel();
    _recognitionTimer = Timer.periodic(
      _recognitionInterval,
      (_) => unawaited(_recognizeFrame()),
    );
  }

  Future<void> _recognizeFrame() async {
    if (_recognizing || !_cameraEnabled) return;
    _recognizing = true;
    try {
      final image = await _captureFrame();
      if (image == null) return;
      final prediction = await _letterRecognizer.recognize(image);
      if (!mounted || !_cameraEnabled) return;

      final matched = prediction.letter == _currentLetter &&
          prediction.confidence >= _minConfidence;
      setState(() {
        _recognitionStatus = prediction.describe();
        _matchStreak = matched ? _matchStreak + 1 : 0;
      });
      if (_matchStreak >= _framesToValidate) _onSignChecked(true);
    } on LetterServerException catch (error) {
      if (mounted) setState(() => _recognitionStatus = error.message);
    } on Exception {
      // The camera was busy or closing: the next tick tries again.
    } finally {
      _recognizing = false;
    }
  }

  Future<Uint8List?> _captureFrame() async {
    final macOSCameraController = _macOSCameraController;
    if (macOSCameraController != null) {
      return (await macOSCameraController.takePicture())?.bytes;
    }

    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        controller.value.isTakingPicture) {
      return null;
    }
    final picture = await controller.takePicture();
    try {
      return await picture.readAsBytes();
    } finally {
      unawaited(deleteCapture(picture.path));
    }
  }

  void _onMacOSCameraInitialized(CameraMacOSController controller) {
    if (!mounted || !_cameraEnabled) {
      unawaited(controller.destroy());
      return;
    }
    _macOSCameraController = controller;
  }

  String _cameraErrorMessage(CameraException error) {
    return switch (error.code) {
      'CameraAccessDenied' ||
      'CameraAccessDeniedWithoutPrompt' ||
      'CameraAccessRestricted' =>
        'Camera access is unavailable. Check your device permissions.',
      'NoCameraAvailable' => error.description ?? 'No camera was found.',
      _ => error.description ?? 'Could not open the camera (${error.code}).',
    };
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final cameraTooltip = _cameraEnabled ? 'Turn camera off' : 'Turn camera on';

    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, pageConstraints) {
            final compact = pageConstraints.maxHeight < 820;

            return Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1000),
                child: Padding(
                  padding: EdgeInsets.fromLTRB(
                    20,
                    compact ? 8 : 20,
                    20,
                    compact ? 12 : 24,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _TrainingHeader(
                        colors: colors,
                        completedLetters: _currentLetterIndex + 1,
                        totalLetters: _letters.length,
                        title: widget.practiceTitle,
                        onBack: () => Navigator.of(context).pop(),
                      ),
                      SizedBox(height: compact ? 8 : 20),
                      if (!compact) ...[
                        Text(
                          'Learn the alphabet',
                          textAlign: TextAlign.center,
                          style: Theme.of(context)
                              .textTheme
                              .headlineSmall
                              ?.copyWith(fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Watch Gragas, then try the sign yourself.',
                          textAlign: TextAlign.center,
                          style: Theme.of(context)
                              .textTheme
                              .bodyMedium
                              ?.copyWith(color: colors.onSurfaceVariant),
                        ),
                        const SizedBox(height: 12),
                      ],
                      Expanded(
                        child: LayoutBuilder(
                          builder: (context, stageConstraints) {
                            final sideBySide = stageConstraints.maxWidth >= 720;
                            final characterPanel = _CharacterPanel(
                              letter: _currentLetter,
                              gragasAnimation: _gragasAnimation,
                              gragasAnimationTrigger: _gragasAnimationTrigger,
                              gragasAvatarKey: _gragasAvatarKey,
                              letterColor: colors.primary,
                              borderColor: colors.outlineVariant,
                            );
                            final cameraPanel = CameraPreviewPanel(
                              isMacOS:
                                  defaultTargetPlatform == TargetPlatform.macOS,
                              controller: _cameraController,
                              isLoading: _cameraLoading,
                              error: _cameraError,
                              onMacOSCameraInitialized:
                                  _onMacOSCameraInitialized,
                            );

                            if (!_cameraEnabled) return characterPanel;

                            if (sideBySide) {
                              return Row(
                                children: [
                                  Expanded(flex: 5, child: characterPanel),
                                  const SizedBox(width: 16),
                                  Expanded(flex: 4, child: cameraPanel),
                                ],
                              );
                            }

                            return Column(
                              children: [
                                Expanded(flex: 6, child: characterPanel),
                                const SizedBox(height: 12),
                                Expanded(flex: 5, child: cameraPanel),
                              ],
                            );
                          },
                        ),
                      ),
                      SizedBox(height: compact ? 8 : 12),
                      _PracticePrompt(
                        foregroundColor: colors.primary,
                        textColor: colors.onSurface,
                        secondaryTextColor: colors.onSurfaceVariant,
                        borderColor: colors.outlineVariant,
                        letter: _currentLetter,
                        cameraEnabled: _cameraEnabled,
                        cameraTooltip: cameraTooltip,
                        cameraError: _cameraError,
                        recognitionStatus:
                            _cameraEnabled ? _recognitionStatus : null,
                        onCameraToggle: _toggleCamera,
                        onSignChecked: _onSignChecked,
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _TrainingHeader extends StatelessWidget {
  const _TrainingHeader({
    required this.colors,
    required this.completedLetters,
    required this.totalLetters,
    required this.title,
    required this.onBack,
  });

  final ColorScheme colors;
  final int completedLetters;
  final int totalLetters;
  final String title;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final progress = completedLetters / totalLetters;

    return Row(
      children: [
        IconButton(
          tooltip: 'Back to menu',
          onPressed: onBack,
          icon: const Icon(Icons.arrow_back_rounded),
        ),
        const SizedBox(width: 4),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.labelMedium?.copyWith(
                      color: colors.onSurfaceVariant,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                    ),
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(value: progress, minHeight: 8),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Text(
          '$completedLetters / $totalLetters',
          style: Theme.of(context)
              .textTheme
              .labelLarge
              ?.copyWith(fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}

class _CharacterPanel extends StatelessWidget {
  const _CharacterPanel({
    required this.letter,
    required this.gragasAnimation,
    required this.gragasAnimationTrigger,
    required this.gragasAvatarKey,
    required this.letterColor,
    required this.borderColor,
  });

  final String letter;
  final GragasAnimation gragasAnimation;
  final int gragasAnimationTrigger;
  final Key gragasAvatarKey;
  final Color letterColor;
  final Color borderColor;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SignPrompt(
          letter: letter,
          letterColor: letterColor,
          borderColor: borderColor,
        ),
        const SizedBox(height: 4),
        Expanded(
          child: GragasAvatar(
            key: gragasAvatarKey,
            animation: gragasAnimation,
            animationTrigger: gragasAnimationTrigger,
          ),
        ),
      ],
    );
  }
}

class _PracticePrompt extends StatelessWidget {
  const _PracticePrompt({
    required this.foregroundColor,
    required this.textColor,
    required this.secondaryTextColor,
    required this.borderColor,
    required this.letter,
    required this.cameraEnabled,
    required this.cameraTooltip,
    required this.cameraError,
    required this.recognitionStatus,
    required this.onCameraToggle,
    required this.onSignChecked,
  });

  final Color foregroundColor;
  final Color textColor;
  final Color secondaryTextColor;
  final Color borderColor;
  final String letter;
  final bool cameraEnabled;
  final String cameraTooltip;
  final String? cameraError;
  final String? recognitionStatus;
  final VoidCallback onCameraToggle;
  final ValueChanged<bool> onSignChecked;

  @override
  Widget build(BuildContext context) {
    final surfaceColor = Theme.of(context).colorScheme.surface;
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: surfaceColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: foregroundColor.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(Icons.front_hand_rounded, color: foregroundColor),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Your turn',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: textColor,
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      'Practice the ASL sign for $letter.',
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(color: secondaryTextColor),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: Tooltip(
                  message: cameraTooltip,
                  child: FilledButton.tonalIcon(
                    onPressed: onCameraToggle,
                    style: FilledButton.styleFrom(
                      minimumSize: const Size.fromHeight(58),
                    ),
                    icon: Icon(
                      cameraEnabled
                          ? Icons.videocam_rounded
                          : Icons.videocam_outlined,
                      size: 24,
                    ),
                    label: Text(
                      cameraEnabled
                          ? 'Turn camera off'
                          : 'Practice with camera (optional)',
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => onSignChecked(true),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(58),
                  ),
                  icon: Icon(
                    Icons.arrow_forward_rounded,
                    size: 24,
                  ),
                  label: const Text('Done'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: TextButton(
              onPressed: () => onSignChecked(false),
              style: TextButton.styleFrom(
                minimumSize: const Size(0, 30),
                padding: EdgeInsets.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                visualDensity: VisualDensity.compact,
              ),
              child: const Text('I need more practice'),
            ),
          ),
          if (recognitionStatus != null && cameraError == null) ...[
            const SizedBox(height: 8),
            Text(
              recognitionStatus!,
              textAlign: TextAlign.center,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: secondaryTextColor),
            ),
          ],
          if (cameraError != null) ...[
            const SizedBox(height: 8),
            Text(
              cameraError!,
              textAlign: TextAlign.center,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.error),
            ),
          ],
        ],
      ),
    );
  }
}
