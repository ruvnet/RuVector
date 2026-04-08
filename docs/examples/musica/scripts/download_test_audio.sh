#!/bin/bash
# Download public domain audio files for Musica evaluation
# Sources: ESC-50 (CC-BY-NC 3.0), Signalogic, SampleLib, exaile
set -e
AUDIO_DIR="$(dirname "$0")/../test_audio"
mkdir -p "$AUDIO_DIR" && cd "$AUDIO_DIR"

echo "Downloading ESC-50 environmental sounds..."
curl -s -o rain.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-17367-A-10.wav"
curl -s -o birds.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100038-A-14.wav"
curl -s -o clapping.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-104089-A-22.wav"
curl -s -o laughing.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-1791-A-26.wav"
curl -s -o dog.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100032-A-0.wav"
curl -s -o church_bells.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-13571-A-46.wav"

echo "Downloading speech samples..."
curl -s -o speech_male.wav "https://www.signalogic.com/melp/EngSamples/Orig/male.wav"
curl -s -o speech_female.wav "https://www.signalogic.com/melp/EngSamples/Orig/female.wav"

echo "Downloading music..."
curl -s -o music_6s.wav "https://samplelib.com/wav/sample-6s.wav"

echo "Downloading test tone..."
curl -s -o noise_tone.wav "https://raw.githubusercontent.com/exaile/exaile-test-files/master/noise_tone.wav"

echo "Downloaded $(ls *.wav | wc -l) WAV files:"
ls -lh *.wav
