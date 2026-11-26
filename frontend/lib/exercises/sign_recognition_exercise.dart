import 'dart:async';

import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../services/capture_cleanup.dart';
import '../services/letter_recognizer.dart';
import '../widgets/camera_preview_panel.dart';
import 'exercise.dart';

class SignRecognitionExercise extends Exercise {
  const SignRecognitionExercise({
    required this.targetLetter,
    required this.onResult,
    this.resetToken = 0,
    super.key,
  });

  final String targetLetter;
  final ValueChanged<bool> onResult;
  final Object resetToken;

  @override
  String get title => 'Sign Recognition';

  @override
  Widget build(BuildContext context) => _SignRecognitionCamera(
        targetLetter: targetLetter,
        onResult: onResult,
        resetToken: resetToken,
      );
}

class _SignRecognitionCamera extends StatefulWidget {
  const _SignRecognitionCamera({
    required this.targetLetter,
    required this.onResult,
    required this.resetToken,
  });

  final String targetLetter;
  final ValueChanged<bool> onResult;
  final Object resetToken;

  @override
  State<_SignRecognitionCamera> createState() => _SignRecognitionCameraState();
}

class _SignRecognitionCameraState extends State<_SignRecognitionCamera>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  CameraMacOSController? _macOSCameraController;
  final _letterRecognizer = LetterRecognizer();
  Timer? _recognitionTimer;
  bool _cameraEnabled = false;
  bool _cameraLoading = false;
  bool _recognizing = false;
  bool _resultSubmitted = false;
  int _matchStreak = 0;
  String? _cameraError;
  String? _recognitionStatus;
  String? _lastServerError;

  static const _framesToValidate = 3;
  static const _minConfidence = 0.8;
  static const _recognitionInterval = Duration(milliseconds: 400);

  String get _targetLetter => widget.targetLetter.trim().toUpperCase();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) unawaited(_toggleCamera());
    });
  }

  @override
  void didUpdateWidget(covariant _SignRecognitionCamera oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.targetLetter != oldWidget.targetLetter ||
        widget.resetToken != oldWidget.resetToken) {
      _resultSubmitted = false;
      _matchStreak = 0;
      _recognitionStatus = null;
      if (_cameraEnabled) _startRecognition();
    }
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
        _lastServerError = null;
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
    if (_recognizing || !_cameraEnabled || _resultSubmitted) return;
    _recognizing = true;
    try {
      final image = await _captureFrame();
      if (image == null) return;
      final prediction = await _letterRecognizer.recognize(image);
      if (!mounted || !_cameraEnabled || _resultSubmitted) return;

      _lastServerError = null;
      final matched = prediction.letter == _targetLetter &&
          prediction.confidence >= _minConfidence;
      setState(() {
        _recognitionStatus = prediction.describe();
        _matchStreak = matched ? _matchStreak + 1 : 0;
      });
      if (_matchStreak >= _framesToValidate) _submitResult(true);
    } on LetterServerException catch (error) {
      if (!mounted) return;
      setState(() => _recognitionStatus = null);
      if (_lastServerError != error.message) {
        _lastServerError = error.message;
        final colors = Theme.of(context).colorScheme;
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(
            SnackBar(
              backgroundColor: colors.error,
              content:
                  Text(error.message, style: TextStyle(color: colors.onError)),
            ),
          );
      }
    } on Exception {
      // The camera was busy or closing; the next tick tries again.
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

  void _submitResult(bool result) {
    if (_resultSubmitted) return;
    _resultSubmitted = true;
    _recognitionTimer?.cancel();
    _recognitionTimer = null;
    widget.onResult(result);
  }

  @override
  Widget build(BuildContext context) {
    final cameraTooltip = _cameraEnabled ? 'Close camera' : 'Open camera';
    final cameraToggle = Tooltip(
      message: cameraTooltip,
      child: IconButton.filledTonal(
        onPressed: _resultSubmitted || _cameraLoading ? null : _toggleCamera,
        icon: Icon(
          _cameraEnabled ? Icons.close_rounded : Icons.videocam_rounded,
        ),
      ),
    );

    return Column(
      children: [
        Expanded(
          child: _cameraEnabled || _cameraLoading || _cameraError != null
              ? CameraPreviewPanel(
                  isMacOS: defaultTargetPlatform == TargetPlatform.macOS,
                  controller: _cameraController,
                  isLoading: _cameraLoading,
                  error: _cameraError,
                  onMacOSCameraInitialized: _onMacOSCameraInitialized,
                  topRightAction: cameraToggle,
                )
              : Center(
                  child: CameraPreviewPanel(
                    isMacOS: false,
                    controller: null,
                    isLoading: false,
                    error: null,
                    isBlank: true,
                    onMacOSCameraInitialized: (_) {},
                    topRightAction: cameraToggle,
                  ),
                ),
        ),
        if (_recognitionStatus != null && _cameraError == null)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(_recognitionStatus!, textAlign: TextAlign.center),
          ),
        Row(
          children: [
            Expanded(
              child: TextButton.icon(
                onPressed: _resultSubmitted ? null : () => _submitResult(true),
                icon: const Icon(Icons.arrow_forward_rounded),
                label: const Text("Can't open camera right now."),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
