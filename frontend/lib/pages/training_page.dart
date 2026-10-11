import 'dart:async';

import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../widgets/gragas_avatar.dart';

class TrainingPage extends StatefulWidget {
  const TrainingPage({super.key});

  @override
  State<TrainingPage> createState() => _TrainingPageState();
}

class _TrainingPageState extends State<TrainingPage>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  CameraMacOSController? _macOSCameraController;
  bool _cameraEnabled = false;
  bool _cameraLoading = false;
  int _currentLetterIndex = 0;
  bool _trainingComplete = false;
  GragasAnimation _gragasAnimation = GragasAnimation.idle;
  int _gragasAnimationTrigger = 0;
  String? _cameraError;

  String get _currentLetter => String.fromCharCode(65 + _currentLetterIndex);

  void _onSignChecked(bool signCompleted) {
    if (!signCompleted || _trainingComplete) return;

    setState(() {
      _gragasAnimation = GragasAnimation.drinkBarrel;
      _gragasAnimationTrigger++;
      if (_currentLetterIndex == 25) {
        _trainingComplete = true;
      } else {
        _currentLetterIndex++;
      }
    });
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
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
    if (mounted) {
      setState(() {
        _cameraEnabled = false;
        _cameraLoading = false;
        _cameraController = null;
        _macOSCameraController = null;
        _cameraError = null;
      });
    }
    await cameraController?.dispose();
    await macOSCameraController?.destroy();
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
                        completedLetters: _trainingComplete
                            ? 26
                            : _currentLetterIndex + 1,
                        onBack: () => Navigator.of(context).pop(),
                      ),
                      SizedBox(height: compact ? 8 : 20),
                      if (!compact) ...[
                        Text(
                          'Learn the alphabet',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.headlineSmall
                              ?.copyWith(fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Watch Gragas, then try the sign yourself.',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodyMedium
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
                              letterColor: colors.primary,
                              borderColor: colors.outlineVariant,
                            );
                            final cameraPanel = _CameraPreviewPanel(
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
                        trainingComplete: _trainingComplete,
                        cameraEnabled: _cameraEnabled,
                        cameraTooltip: cameraTooltip,
                        cameraError: _cameraError,
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
    required this.onBack,
  });

  final ColorScheme colors;
  final int completedLetters;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final progress = completedLetters / 26;

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
                'ALPHABET PRACTICE',
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
          '$completedLetters / 26',
          style: Theme.of(context).textTheme.labelLarge
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
    required this.letterColor,
    required this.borderColor,
  });

  final String letter;
  final GragasAnimation gragasAnimation;
  final int gragasAnimationTrigger;
  final Color letterColor;
  final Color borderColor;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _SpeechBubble(
          letter: letter,
          letterColor: letterColor,
          borderColor: borderColor,
        ),
        const SizedBox(height: 4),
        Expanded(
          child: GragasAvatar(
            animation: gragasAnimation,
            animationTrigger: gragasAnimationTrigger,
          ),
        ),
      ],
    );
  }
}

class _SpeechBubble extends StatelessWidget {
  const _SpeechBubble({
    required this.letter,
    required this.letterColor,
    required this.borderColor,
  });

  final String letter;
  final Color letterColor;
  final Color borderColor;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
          decoration: BoxDecoration(
            color: Colors.white,
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
                style: Theme.of(context).textTheme.headlineMedium
                    ?.copyWith(color: letterColor, fontWeight: FontWeight.w900),
              ),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  'Can you make this sign?',
                  style: Theme.of(context).textTheme.bodyLarge
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
              color: Colors.white,
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

class _PracticePrompt extends StatelessWidget {
  const _PracticePrompt({
    required this.foregroundColor,
    required this.textColor,
    required this.secondaryTextColor,
    required this.borderColor,
    required this.letter,
    required this.trainingComplete,
    required this.cameraEnabled,
    required this.cameraTooltip,
    required this.cameraError,
    required this.onCameraToggle,
    required this.onSignChecked,
  });

  final Color foregroundColor;
  final Color textColor;
  final Color secondaryTextColor;
  final Color borderColor;
  final String letter;
  final bool trainingComplete;
  final bool cameraEnabled;
  final String cameraTooltip;
  final String? cameraError;
  final VoidCallback onCameraToggle;
  final ValueChanged<bool> onSignChecked;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
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
                      trainingComplete
                          ? 'You completed the alphabet!'
                          : 'Practice the ASL sign for $letter.',
                      style: Theme.of(context).textTheme.bodyMedium
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
                  onPressed: trainingComplete
                      ? null
                      : () => onSignChecked(true),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(58),
                  ),
                  icon: Icon(
                    trainingComplete
                        ? Icons.check_circle_rounded
                        : Icons.arrow_forward_rounded,
                    size: 24,
                  ),
                  label: Text(trainingComplete ? 'Completed' : 'Done'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (cameraError != null) ...[
            const SizedBox(height: 8),
            Text(
              cameraError!,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.error),
            ),
          ],
        ],
      ),
    );
  }
}

class _CameraPreviewPanel extends StatelessWidget {
  const _CameraPreviewPanel({
    required this.isMacOS,
    required this.controller,
    required this.isLoading,
    required this.error,
    required this.onMacOSCameraInitialized,
  });

  final bool isMacOS;
  final CameraController? controller;
  final bool isLoading;
  final String? error;
  final ValueChanged<CameraMacOSController> onMacOSCameraInitialized;

  @override
  Widget build(BuildContext context) {
    final preview = switch (true) {
      _ when error != null => _CameraStatus(
        icon: Icons.videocam_off_outlined,
        message: error!,
      ),
      _ when isLoading => const _CameraStatus(
        icon: Icons.videocam_outlined,
        message: 'Opening camera…',
        showProgress: true,
      ),
      _ when isMacOS => CameraMacOSView(
        cameraMode: CameraMacOSMode.photo,
        enableAudio: false,
        fit: BoxFit.cover,
        onCameraInizialized: onMacOSCameraInitialized,
        onCameraLoading: (error) => _CameraStatus(
          icon: Icons.videocam_off_outlined,
          message: error == null
              ? 'Requesting camera access…'
              : 'Could not open the camera: $error',
          showProgress: error == null,
        ),
      ),
      _ when controller?.value.isInitialized == true => CameraPreview(
        controller!,
      ),
      _ => const _CameraStatus(
        icon: Icons.videocam_outlined,
        message: 'Opening camera…',
        showProgress: true,
      ),
    };

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480, maxHeight: 480),
        child: AspectRatio(
          aspectRatio: 4 / 3,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: ColoredBox(
              color: Colors.black,
              child: Center(child: preview),
            ),
          ),
        ),
      ),
    );
  }
}

class _CameraStatus extends StatelessWidget {
  const _CameraStatus({
    required this.icon,
    required this.message,
    this.showProgress = false,
  });

  final IconData icon;
  final String message;
  final bool showProgress;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showProgress)
            const CircularProgressIndicator()
          else
            Icon(icon, color: Colors.white, size: 48),
          const SizedBox(height: 16),
          Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white),
          ),
        ],
      ),
    );
  }
}
