// Deletes a picture file once its bytes have been read.
//
// camera_windows saves every takePicture() in the user's Pictures folder, and
// the camera plugins on Android and iOS write theirs to a cache folder. The
// letter recognition takes a few pictures per second, so they must not pile
// up. dart:io does not exist on the web, hence the conditional import.
export 'capture_cleanup_stub.dart'
    if (dart.library.io) 'capture_cleanup_io.dart';
