import 'dart:async';

import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:frontend/widgets/camera_preview_panel.dart';
import 'package:frontend/widgets/sign_prompt.dart';

import 'training_exercise.dart';

class SignRecognitionExercise extends TrainingExercise {
  const SignRecognitionExercise({this.letter = 'A', super.key});

  final String letter;

  @override
  String get title => 'Sign Recognition';

  @override
  Widget build(BuildContext context) =>
      _SignRecognitionExerciseBody(letter: letter);
}

class _SignRecognitionExerciseBody extends StatefulWidget {
  const _SignRecognitionExerciseBody({required this.letter});

  final String letter;

  @override
  State<_SignRecognitionExerciseBody> createState() =>
      _SignRecognitionExerciseBodyState();
}

class _SignRecognitionExerciseBodyState
    extends State<_SignRecognitionExerciseBody> with WidgetsBindingObserver {
  CameraController? _cameraController;
  CameraMacOSController? _macOSCameraController;
  bool _cameraEnabled = false;
  bool _cameraLoading = false;
  String? _cameraError;

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
    final isMacOS = defaultTargetPlatform == TargetPlatform.macOS;
    final cameraTooltip = _cameraEnabled ? 'Turn camera off' : 'Turn camera on';

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          SignPrompt(
            letter: widget.letter,
            prompt: 'Can you make this sign?',
            letterColor: colors.primary,
            borderColor: colors.outlineVariant,
          ),
          const SizedBox(height: 16),
          Expanded(
            child: _cameraEnabled || _cameraLoading || _cameraError != null
                ? CameraPreviewPanel(
                    isMacOS: isMacOS,
                    controller: _cameraController,
                    isLoading: _cameraLoading,
                    error: _cameraError,
                    onMacOSCameraInitialized: _onMacOSCameraInitialized,
                  )
                : _CameraOffPlaceholder(color: colors.onSurfaceVariant),
          ),
          const SizedBox(height: 12),
          Tooltip(
            message: cameraTooltip,
            child: FilledButton.tonalIcon(
              onPressed: _toggleCamera,
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(54),
              ),
              icon: Icon(
                _cameraEnabled
                    ? Icons.videocam_rounded
                    : Icons.videocam_outlined,
              ),
              label: Text(
                _cameraEnabled ? 'Turn camera off' : 'Practice with camera',
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CameraOffPlaceholder extends StatelessWidget {
  const _CameraOffPlaceholder({required this.color});

  final Color color;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.videocam_outlined, size: 48, color: color),
          const SizedBox(height: 8),
          Text(
            'Turn on the camera to practice',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: color,
                ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
