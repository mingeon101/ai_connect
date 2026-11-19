
"""Gemini HTTP-to-HTTPS relay for Render and ESP32."""

import os
import http.server
import base64
import io
import json
import socket
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

# Render automatically provides PORT.
PORT = int(os.environ.get("PORT", "8080"))

GEMINI_MODEL = 'gemini-3.5-flash-lite'
TARGET = (
    'https://generativelanguage.googleapis.com/v1beta/models/'
    + GEMINI_MODEL + ':generateContent'
)

TTS_MODEL = 'gemini-3.8-flash-tts'
TTS_VOICE = 'Kore'
TTS_TARGET = 'https://generativelanguage.googleapis.com/v1beta/interactions'

PCM_RATE = 8000
MAX_SPEECH_SECONDS = 45

DISPLAY_INSTRUCTION = (
    '또한 display_text에는 답변의 핵심을 영어 ASCII 문자만 사용하여 '
    '60자 이내의 짧은 문장으로 쓰세요. 5x5 LED에서 읽을 수 있도록 '
    '한글, 이모지, 마크다운, 줄바꿈은 사용하지 마세요.'
)


def display_text(value):
    if not isinstance(value, str):
        return 'SEE SERIAL MONITOR'

    value = ' '.join(value.split())
    value = ''.join(c for c in value if 32 <= ord(c) <= 126)

    return value.strip()[:60] or 'SEE SERIAL MONITOR'


def make_text_request(data):
    request = json.loads(data.decode('utf-8'))

    if not isinstance(request, dict) or not isinstance(
        request.get('contents'), list
    ):
        raise ValueError('Missing contents')

    previous = request.get(
        'systemInstruction', {}
    ).get('parts', [])

    request['systemInstruction'] = {
        'parts': list(previous) + [{
            'text': (
                'answer에는 질문에 대한 답변을 한국어로 작성하세요. '
                + DISPLAY_INSTRUCTION
            )
        }]
    }

    config = request.setdefault('generationConfig', {})
    config['responseMimeType'] = 'application/json'
    config['responseSchema'] = {
        'type': 'OBJECT',
        'properties': {
            'answer': {'type': 'STRING'},
            'display_text': {'type': 'STRING'}
        },
        'required': ['answer', 'display_text']
    }

    return json.dumps(
        request, ensure_ascii=False
    ).encode('utf-8')


def structured_result(body):
    response = json.loads(body.decode('utf-8'))
    candidates = response.get('candidates', [])

    if not candidates:
        raise ValueError('No Gemini candidate')

    text = ''.join(
        part.get('text', '')
        for part in candidates[0].get(
            'content', {}
        ).get('parts', [])
        if not part.get('thought')
    )

    result = json.loads(text)

    if not isinstance(result, dict) or not isinstance(
        result.get('answer'), str
    ):
        raise ValueError('Missing answer')

    return response, result


def text_reply(body):
    response, result = structured_result(body)

    response['candidates'][0]['content']['parts'] = [
        {'text': result['answer']}
    ]

    response['display_text'] = display_text(
        result.get('display_text')
    )

    encoded = json.dumps(
        response, ensure_ascii=False
    ).encode('utf-8')

    if len(encoded) > 8192:
        raise ValueError('Text answer exceeds device buffer')

    return encoded


def wav_to_pcm8(data):
    """Convert 16-bit PCM WAV to unsigned 8-bit mono 8kHz."""

    try:
        with wave.open(io.BytesIO(data), 'rb') as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            count = wav.getnframes()

            if (
                wav.getsampwidth() != 2
                or wav.getcomptype() != 'NONE'
                or channels not in (1, 2)
                or rate not in (8000, 16000, 24000, 32000, 48000)
                or not count
                or count > rate * MAX_SPEECH_SECONDS
            ):
                raise ValueError('Unsupported or overlong TTS WAV')

            raw = wav.readframes(count)

            if len(raw) != count * channels * 2:
                raise ValueError('Incomplete TTS WAV')

    except (wave.Error, EOFError) as exc:
        raise ValueError('TTS response is not PCM WAV') from exc

    step = rate // PCM_RATE
    pcm = bytearray()

    values = struct.unpack(
        '<{}h'.format(count * channels), raw
    )

    for start in range(0, count - step + 1, step):
        block = values[
            start * channels:(start + step) * channels
        ]

        signed = sum(block) // len(block)

        pcm.append(
            max(0, min(255, (signed + 32768) // 256))
        )

    return bytes(pcm)


def synthesize_speech(text, key):
    request_body = {
        'model': TTS_MODEL,
        'input': [{
            'type': 'user_input',
            'content': [{
                'type': 'text',
                'text': text
            }]
        }],
        'response_format': {
            'type': 'audio',
            'mime_type': 'audio/wav',
            'sample_rate': 24000
        },
        'generation_config': {
            'speech_config': [{
                'voice': TTS_VOICE
            }]
        }
    }

    request = urllib.request.Request(
        TTS_TARGET,
        data=json.dumps(
            request_body, ensure_ascii=False
        ).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'x-goog-api-key': key,
            'Api-Revision': '2026-05-20'
        },
        method='POST'
    )

    with urllib.request.urlopen(
        request, timeout=60
    ) as response:
        body = response.read(4000001)

    if len(body) > 4000000:
        raise ValueError('TTS response too large')

    result = json.loads(body.decode('utf-8'))
    result = result.get('interaction', result)

    blocks = []

    for step in result.get('steps', []):
        if step.get('type') == 'model_output':
            blocks.extend(
                part
                for part in step.get('content', [])
                if part.get('type') == 'audio'
                and part.get('data')
            )

    if not blocks:
        raise ValueError('No inline audio in TTS response')

    return wav_to_pcm8(
        base64.b64decode(
            blocks[-1]['data'], validate=True
        )
    )


def make_audio_request(data):
    """Validate audio and ask Gemini for transcript and answer."""

    try:
        with wave.open(io.BytesIO(data), 'rb') as wav:
            rate = wav.getframerate()
            count = wav.getnframes()

            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or rate not in (8000, 16000)
                or wav.getcomptype() != 'NONE'
                or not 0 < count <= rate * 4
            ):
                raise ValueError(
                    'Use mono 16-bit PCM WAV, '
                    '8/16 kHz, up to 4 seconds'
                )

            if len(wav.readframes(count)) != count * 2:
                raise ValueError('Incomplete WAV samples')

    except (wave.Error, EOFError) as exc:
        raise ValueError('Invalid WAV') from exc

    prompt = (
        '녹음된 사용자의 말을 한국어로 정확하게 받아쓰고 '
        '그 질문에 한국어 3문장 이내로 답하세요. '
        'transcript에는 실제로 들린 말만 쓰세요. '
        '말소리가 없거나 알아들을 수 없으면 '
        'transcript를 빈 문자열로 쓰고 '
        'answer에는 다시 말해달라고 요청하세요. '
        '녹음에 없는 질문을 만들어내지 마세요. '
        + DISPLAY_INSTRUCTION
    )

    request_body = {
        'contents': [{
            'role': 'user',
            'parts': [
                {'text': prompt},
                {
                    'inlineData': {
                        'mimeType': 'audio/wav',
                        'data': base64.b64encode(
                            data
                        ).decode('ascii')
                    }
                }
            ]
        }],
        'generationConfig': {
            'maxOutputTokens': 1024,
            'responseMimeType': 'application/json',
            'responseSchema': {
                'type': 'OBJECT',
                'properties': {
                    'transcript': {'type': 'STRING'},
                    'answer': {'type': 'STRING'},
                    'display_text': {'type': 'STRING'}
                },
                'required': [
                    'transcript',
                    'answer',
                    'display_text'
                ]
            }
        }
    }

    return json.dumps(
        request_body, ensure_ascii=False
    ).encode('utf-8')


def audio_reply(body):
    _, result = structured_result(body)

    if not isinstance(result, dict) or not all(
        isinstance(result.get(k), str)
        for k in ('transcript', 'answer')
    ):
        raise ValueError('Missing transcript or answer')

    clean = {
        'transcript': result['transcript'],
        'answer': result['answer'],
        'display_text': display_text(
            result.get('display_text')
        )
    }

    encoded = json.dumps(
        clean, ensure_ascii=False
    ).encode('utf-8')

    if len(encoded) > 8192:
        raise ValueError('Audio answer exceeds device buffer')

    return encoded


class GeminiRelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def reply(self, status, body):
        self.send_response(status)
        self.send_header(
            'Content-Type',
            'application/json; charset=utf-8'
        )
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def error_json(self, status, message):
        body = json.dumps(
            {'error': {'message': message}},
            ensure_ascii=False
        ).encode('utf-8')

        self.reply(status, body)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path

        if path in ('/', '/health'):
            body = json.dumps({
                'status': 'ok',
                'service': 'Gemini Relay',
                'model': GEMINI_MODEL
            }).encode('utf-8')

            self.reply(200, body)
        else:
            self.error_json(404, 'Not found')

    def do_POST(self):
        try:
            if self.headers.get('Transfer-Encoding'):
                self.error_json(
                    400,
                    'Use Content-Length, not Transfer-Encoding'
                )
                return

            path = urllib.parse.urlsplit(self.path).path

            if path not in ('/', '/audio', '/tts'):
                self.error_json(404, 'Not found')
                return

            audio = path == '/audio'
            length = int(
                self.headers.get('Content-Length', '0')
            )

            maximum = (
                128044 if audio
                else 16384
            )

            if not 0 < length <= maximum:
                self.error_json(
                    400,
                    'Invalid Content-Length (maximum {})'.format(
                        maximum
                    )
                )
                return

            data = self.rfile.read(length)

            if len(data) != length:
                self.error_json(
                    400, 'Incomplete request body'
                )
                return

            if path == '/tts':
                self.handle_tts(data)
                return

            if audio:
                if self.headers.get_content_type() != 'audio/wav':
                    self.error_json(
                        415,
                        'Audio endpoint requires audio/wav'
                    )
                    return

                data = make_audio_request(data)

            else:
                data = make_text_request(data)

            query = urllib.parse.parse_qs(
                urllib.parse.urlsplit(self.path).query
            )

            key = (
                self.headers.get('x-goog-api-key', '')
                or os.environ.get('GEMINI_API_KEY', '')
                or query.get('key', [''])[0]
            )

            if not key:
                self.error_json(
                    401, 'Missing Gemini API key'
                )
                return

            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': key
            }

            request = urllib.request.Request(
                TARGET,
                data=data,
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(
                request, timeout=60
            ) as response:
                body = response.read(65537)
                status = response.status

            if len(body) > 65536:
                self.error_json(
                    502, 'Gemini response too large'
                )
                return

            if status == 200:
                try:
                    body = (
                        audio_reply(body)
                        if audio
                        else text_reply(body)
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError
                ):
                    self.error_json(
                        502,
                        'Gemini did not return a valid answer'
                    )
                    return

            self.reply(status, body)

            print(
                '[SUCCESS] Gemini status:', status,
                flush=True
            )

        except urllib.error.HTTPError as exc:
            body = exc.read(65536)
            self.reply(exc.code, body)

            print(
                '[HTTP ERROR] Gemini status:',
                exc.code,
                flush=True
            )

        except (ValueError, UnicodeError):
            self.error_json(
                400,
                'Invalid JSON or Content-Length'
            )

        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError
        ):
            self.error_json(
                502,
                'Gemini network request failed or timed out'
            )

        except (BrokenPipeError, ConnectionResetError):
            pass

        except Exception as exc:
            print(
                '[ERROR]',
                type(exc).__name__,
                flush=True
            )

            self.error_json(
                500, 'Relay request failed'
            )

    def speech_reply(self, pcm):
        self.send_response(200)

        self.send_header(
            'Content-Type',
            'application/octet-stream'
        )

        self.send_header(
            'Content-Length',
            str(len(pcm))
        )

        self.send_header(
            'X-Audio-Format',
            'u8-mono-8000'
        )

        self.send_header('Connection', 'close')
        self.end_headers()

        initial = min(8192, len(pcm))

        self.wfile.write(pcm[:initial])
        self.wfile.flush()

        started = time.monotonic()

        for offset in range(initial, len(pcm), 4096):
            wait = (
                (offset - initial) / PCM_RATE
                - (time.monotonic() - started)
            )

            if wait > 0:
                time.sleep(wait)

            self.wfile.write(
                pcm[offset:offset + 4096]
            )
            self.wfile.flush()

        self.close_connection = True

    def handle_tts(self, data):
        obj = json.loads(data.decode('utf-8'))

        text = (
            obj.get('text', '')
            if isinstance(obj, dict)
            else ''
        )

        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 1200
        ):
            self.error_json(
                400,
                'TTS text must contain 1..1200 characters'
            )
            return

        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(self.path).query
        )

        key = (
            self.headers.get('x-goog-api-key', '')
            or os.environ.get('GEMINI_API_KEY', '')
            or query.get('key', [''])[0]
        )

        if not key:
            self.error_json(
                401, 'Missing Gemini API key'
            )
            return

        try:
            pcm = synthesize_speech(text, key)

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError
        ):
            self.error_json(
                502,
                'TTS did not return supported WAV audio'
            )
            return

        self.speech_reply(pcm)

        print(
            '[SUCCESS] TTS samples:',
            len(pcm),
            flush=True
        )

    def log_message(self, fmt, *args):
        # Do not log API keys in query strings.
        print(
            '[{}] request processed'.format(
                self.address_string()
            ),
            flush=True
        )


if __name__ == '__main__':
    server = http.server.ThreadingHTTPServer(
        ('0.0.0.0', PORT),
        GeminiRelayHandler
    )

    server.daemon_threads = True

    print(
        'Gemini Relay Server running on port {}'.format(
            PORT
        ),
        flush=True
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print('Shutting down server...', flush=True)

    finally:
        server.server_close()
