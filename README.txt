Digonto Audio 1.0
=================
Record, edit and clean up sound. Layout like Adobe Audition, made easy. Free on Ubuntu, macOS and Windows.

Open: WAV, MP3, FLAC, OGG, Opus, M4A, AAC, AIFF, WMA and the sound of most video files (MP4, MOV, MKV...).
Drop files on the window. Every sound opens in its own tab.

Save: Ctrl+S saves a .paud project (our own format: everything stays exact, markers included).
Export (Ctrl+E): WAV 16/24/32-bit, AIFF, MP3, OGG Vorbis, Opus, FLAC, M4A (AAC or lossless), WMA, AC3.
You can export everything, only the selection, or every part between the markers as separate files.

Edit by milliseconds: drag to select, or Edit > Cut by time and type exact times (0:12.345 or 250ms).
Zoom all the way to single samples. Arrow keys move the cursor 10 ms (Alt = 1 ms, Ctrl = 1 s). "Snap to zero" makes click-free cuts.
Cut, copy, paste, delete, crop, silence, insert silence, remove silence at the ends, shorten long pauses, markers.

Equalizer: 10 bands in the right panel, with presets (voice clarity, bass boost ...), Preview and Apply.
Also: high-pass, low-pass, bell, shelves.

Echo and space: Echo, Reverb, Chorus / Flanger, Tremolo, Distortion.
Volume: Amplify, Make quiet audio louder (with soft peak protection), Normalize, Even out voice, Fades, Compressor, Limiter.
Clean up: Remove background noise (learn the noise from a selection, or automatic), Remove hum, Noise gate, Remove clicks.
Speed and pitch: Pitch (keep speed), Tempo (keep pitch), Speed.
Stereo: mono, pan, width, swap, Remove the voice (karaoke), Keep only the voice.
Split into voice and music: makes two new sounds. It works from where each sound sits between the speakers, so it is
best with a stereo song where the singer is in the middle. It is not artificial intelligence: some voice can stay in the
music and the other way round.

Record from the microphone (red button). On macOS allow the microphone when asked.
Every effect works on the selection (or everything if nothing is selected) and has Preview.
Spectrogram view shows the frequencies; "dB scale" shows quiet details.

Locked libraries (SHA-256): PySide6-Essentials, numpy, and a private ffmpeg (imageio-ffmpeg) for MP3 / OGG / FLAC / M4A.
Not in this version: multitrack mixing (later), VST plug-ins, video.
