# Local Audio Transcription

A FastAPI service that transcribes WAV and MP3 files locally using faster-whisper (`small`) and Silero VAD. Returns the transcript and segment timestamps from the original recording. Runs on CPU by default.

## Setup

Requires **Python 3.11** and **FFmpeg** (including `ffprobe`) on your `PATH`. Windows also requires the Microsoft Visual C++ runtime. Internet access is needed for dependency installation and the first model download.

Run the following commands from the project root. Skip copying `.env.example` if `.env` already exists.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

Install Python 3.11 and FFmpeg using your package manager, then run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

If FFmpeg is not on `PATH`, set `FFMPEG_PATH` and `FFPROBE_PATH` in `.env` to the executable paths. The project has been tested on Windows.

## Run

With the virtual environment active:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The model downloads on first startup and is cached for later runs. Wait for startup to finish, then open [Swagger UI](http://127.0.0.1:8000/docs) to upload a file and try the API.

| Endpoint | Purpose |
|---|---|
| `POST /transcribe` | Upload a WAV or MP3 file |
| `GET /health` | Check that the service is running |
| `GET /ready` | Check that the service is ready to transcribe |

## Example Request

Sample recordings are included in `sample_audio/`. See the [sample details and attribution](sample_audio/README.md).

**Windows PowerShell**

```powershell
curl.exe --max-time 120 -X POST http://127.0.0.1:8000/transcribe -F "file=@sample_audio/2035-147960-0013.mp3" -F "language=en"
```

**Linux / macOS**

```bash
curl --max-time 120 -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@sample_audio/2035-147960-0013.mp3" \
  -F "language=en"
```

The `language` field is optional; omit it for automatic detection. The JSON response includes `status`, `language`, `duration_seconds`, `text`, and `segments` with `start`, `end`, and `text` fields.

### Example Response for `A1.wav`

```json
{
  "status": "success",
  "language": "en",
  "duration_seconds": 18.3561875,
  "text": "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.",
  "segments": [
    {
      "start": 0.91,
      "end": 3.91,
      "text": "The stale smell of old beer lingers."
    },
    {
      "start": 3.91,
      "end": 6.91,
      "text": "It takes heat to bring out the odor."
    },
    {
      "start": 6.91,
      "end": 9.91,
      "text": "A cold dip restores health and zest."
    },
    {
      "start": 9.91,
      "end": 12.91,
      "text": "A salt pickle tastes fine with ham."
    },
    {
      "start": 12.91,
      "end": 14.91,
      "text": "Tacos al pastor are my favorite."
    },
    {
      "start": 14.91,
      "end": 17.91,
      "text": "A zestful food is the hot cross bun."
    }
  ]
}
```

## Tests

Run the default test suite:

```bash
python -m pytest -q
```

Run integration tests with FFmpeg installed and the model cached from a previous startup:

```bash
python -m pytest -q -m integration
```

## Docker

```bash
docker build -t local-transcription .
docker run --rm -p 8000:8000 -v whisper-cache:/app/.cache/models local-transcription
```
