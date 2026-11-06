import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

/// What the letter server saw in one camera frame.
class LetterPrediction {
  const LetterPrediction({
    required this.handFound,
    this.letter,
    this.confidence = 0,
  });

  final bool handFound;
  final String? letter;
  final double confidence;

  String describe() => handFound
      ? 'Seen: $letter (${(confidence * 100).round()}%)'
      : 'No hand detected';
}

class LetterServerException implements Exception {
  const LetterServerException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Sends camera frames to the local sign server (backend/server.py), which
/// runs the same pipeline as the Python letter demo.
///
/// The server address defaults to http://127.0.0.1:8000. Override it at build
/// time, e.g. for the Android emulator, which reaches the host at 10.0.2.2:
///   flutter run --dart-define=LETTER_SERVER=http://10.0.2.2:8000
class LetterRecognizer {
  LetterRecognizer({String? serverUrl})
    : _endpoint = Uri.parse(
        serverUrl ??
            const String.fromEnvironment(
              'LETTER_SERVER',
              defaultValue: 'http://127.0.0.1:8000',
            ),
      ).resolve('/letter');

  final Uri _endpoint;
  final http.Client _client = http.Client();

  Future<LetterPrediction> recognize(Uint8List image) async {
    final http.Response response;
    try {
      response = await _client
          .post(
            _endpoint,
            headers: const {'Content-Type': 'application/octet-stream'},
            body: image,
          )
          .timeout(const Duration(seconds: 3));
    } on TimeoutException {
      throw const LetterServerException('Letter server is not responding.');
    } catch (_) {
      throw const LetterServerException(
        'Letter server unreachable. Start backend/server.py.',
      );
    }

    if (response.statusCode != 200) {
      throw LetterServerException(
        'Letter server error (${response.statusCode}).',
      );
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    final handFound = json['hand'] == true;
    return LetterPrediction(
      handFound: handFound,
      letter: handFound ? json['letter'] as String? : null,
      confidence: handFound ? (json['confidence'] as num).toDouble() : 0,
    );
  }

  void close() => _client.close();
}
