import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:flutter/material.dart';

class CameraPreviewPanel extends StatelessWidget {
  const CameraPreviewPanel({
    required this.isMacOS,
    required this.controller,
    required this.isLoading,
    required this.error,
    required this.onMacOSCameraInitialized,
    this.isBlank = false,
    this.topRightAction,
    super.key,
  });

  final bool isMacOS;
  final CameraController? controller;
  final bool isLoading;
  final String? error;
  final bool isBlank;
  final ValueChanged<CameraMacOSController> onMacOSCameraInitialized;
  final Widget? topRightAction;

  @override
  Widget build(BuildContext context) {
    final preview = switch (true) {
      _ when isBlank => const SizedBox.shrink(),
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
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Center(child: preview),
                  if (topRightAction != null)
                    Positioned(
                      top: 8,
                      right: 8,
                      child: topRightAction!,
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
