import 'dart:io';

Future<void> deleteCapture(String path) async {
  try {
    await File(path).delete();
  } on FileSystemException {
    // Already gone, or still locked by the camera: nothing else to do.
  }
}
