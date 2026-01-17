#!/usr/bin/env python3
"""
Voice Sample Preparation Tool für XTTS v2.
Validiert, optimiert und konvertiert Voice Samples für optimale Qualität.

Usage:
    python scripts/prepare_voice_samples.py samples/host.wav
    python scripts/prepare_voice_samples.py samples/  # Alle WAVs im Ordner
"""
import subprocess
import sys
import wave
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tempfile
import shutil


@dataclass
class SampleAnalysis:
    """Analyse-Ergebnis eines Voice Samples."""
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    peak_db: float
    rms_db: float
    silence_ratio: float
    clipping_detected: bool
    issues: list[str]

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def quality_score(self) -> int:
        """Qualitätsscore von 0-100."""
        score = 100

        # Länge
        if self.duration_seconds < 6:
            score -= 30
        elif self.duration_seconds < 10:
            score -= 15
        elif self.duration_seconds > 60:
            score -= 10

        # Lautstärke
        if self.rms_db < -30:
            score -= 20
        elif self.rms_db < -24:
            score -= 10

        # Clipping
        if self.clipping_detected:
            score -= 25

        # Stille
        if self.silence_ratio > 0.3:
            score -= 15

        return max(0, score)


def check_ffmpeg() -> bool:
    """Prüft ob FFmpeg installiert ist."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def analyze_sample(path: Path) -> SampleAnalysis:
    """Analysiert ein Voice Sample."""
    issues = []

    # Basis-Infos mit wave
    try:
        with wave.open(str(path), 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            bit_depth = wav.getsampwidth() * 8
            frames = wav.getnframes()
            duration = frames / sample_rate
    except Exception as e:
        return SampleAnalysis(
            path=path,
            duration_seconds=0,
            sample_rate=0,
            channels=0,
            bit_depth=0,
            peak_db=-100,
            rms_db=-100,
            silence_ratio=1.0,
            clipping_detected=False,
            issues=[f"Kann Datei nicht lesen: {e}"]
        )

    # FFprobe für detaillierte Analyse
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path)
        ], capture_output=True, text=True)
        probe_data = json.loads(result.stdout)
    except Exception:
        probe_data = {}

    # Lautstärke-Analyse mit FFmpeg
    peak_db = -100.0
    rms_db = -100.0
    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(path),
            "-af", "volumedetect",
            "-f", "null", "-"
        ], capture_output=True, text=True)

        for line in result.stderr.split('\n'):
            if 'max_volume' in line:
                peak_db = float(line.split(':')[1].strip().replace(' dB', ''))
            if 'mean_volume' in line:
                rms_db = float(line.split(':')[1].strip().replace(' dB', ''))
    except Exception:
        pass

    # Stille-Erkennung
    silence_ratio = 0.0
    try:
        result = subprocess.run([
            "ffmpeg", "-i", str(path),
            "-af", "silencedetect=noise=-40dB:d=0.5",
            "-f", "null", "-"
        ], capture_output=True, text=True)

        silence_duration = 0.0
        for line in result.stderr.split('\n'):
            if 'silence_duration' in line:
                try:
                    silence_duration += float(line.split(':')[-1].strip())
                except ValueError:
                    pass

        if duration > 0:
            silence_ratio = silence_duration / duration
    except Exception:
        pass

    # Clipping-Erkennung
    clipping_detected = peak_db > -0.5

    # Issues sammeln
    if duration < 6:
        issues.append(f"Zu kurz ({duration:.1f}s) - mindestens 6 Sekunden empfohlen")
    elif duration < 10:
        issues.append(f"Etwas kurz ({duration:.1f}s) - 15-30 Sekunden optimal")
    elif duration > 60:
        issues.append(f"Sehr lang ({duration:.1f}s) - 15-30 Sekunden optimal")

    if sample_rate != 22050:
        issues.append(f"Sample-Rate {sample_rate}Hz - wird auf 22050Hz konvertiert")

    if channels != 1:
        issues.append(f"{channels} Kanäle - wird zu Mono konvertiert")

    if rms_db < -30:
        issues.append(f"Sehr leise ({rms_db:.1f}dB) - Normalisierung empfohlen")
    elif rms_db < -24:
        issues.append(f"Etwas leise ({rms_db:.1f}dB) - Normalisierung empfohlen")

    if clipping_detected:
        issues.append(f"Clipping erkannt ({peak_db:.1f}dB) - Audio ist übersteuert")

    if silence_ratio > 0.3:
        issues.append(f"Viel Stille ({silence_ratio*100:.0f}%) - Sample kürzen")

    return SampleAnalysis(
        path=path,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        peak_db=peak_db,
        rms_db=rms_db,
        silence_ratio=silence_ratio,
        clipping_detected=clipping_detected,
        issues=issues
    )


def optimize_sample(
    input_path: Path,
    output_path: Optional[Path] = None,
    target_sample_rate: int = 22050,
    normalize: bool = True,
    noise_reduction: bool = True,
    trim_silence: bool = True
) -> Path:
    """
    Optimiert ein Voice Sample für XTTS v2.

    Args:
        input_path: Eingabe-Datei
        output_path: Ausgabe-Datei (optional, überschreibt Eingabe wenn None)
        target_sample_rate: Ziel-Sample-Rate (22050 für XTTS v2)
        normalize: Lautstärke normalisieren
        noise_reduction: Rauschen reduzieren
        trim_silence: Stille am Anfang/Ende entfernen

    Returns:
        Pfad zur optimierten Datei
    """
    if output_path is None:
        output_path = input_path

    # Temporäre Datei für Zwischenschritte
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        current_input = input_path

        # Filter-Chain aufbauen
        filters = []

        # 1. Mono konvertieren
        filters.append("aformat=channel_layouts=mono")

        # 2. Stille trimmen
        if trim_silence:
            filters.append("silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB")
            filters.append("areverse")
            filters.append("silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB")
            filters.append("areverse")

        # 3. Noise Reduction (Highpass + Lowpass)
        if noise_reduction:
            filters.append("highpass=f=80")
            filters.append("lowpass=f=12000")
            # Sanfte Noise Gate
            filters.append("agate=threshold=-40dB:ratio=2:attack=10:release=100")

        # 4. Normalisierung
        if normalize:
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        filter_chain = ",".join(filters)

        # FFmpeg ausführen
        cmd = [
            "ffmpeg", "-y",
            "-i", str(current_input),
            "-af", filter_chain,
            "-ar", str(target_sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(temp_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Fehler: {result.stderr}")

        # Zur Ausgabe kopieren
        shutil.move(str(temp_path), str(output_path))

        return output_path

    finally:
        # Aufräumen
        if temp_path.exists():
            temp_path.unlink()


def print_analysis(analysis: SampleAnalysis):
    """Gibt die Analyse formatiert aus."""
    print(f"\n{'='*60}")
    print(f"Datei: {analysis.path.name}")
    print(f"{'='*60}")

    # Basis-Info
    print(f"\nTechnische Daten:")
    print(f"  Dauer:       {analysis.duration_seconds:.1f} Sekunden")
    print(f"  Sample-Rate: {analysis.sample_rate} Hz")
    print(f"  Kanäle:      {analysis.channels}")
    print(f"  Bit-Tiefe:   {analysis.bit_depth} bit")

    # Lautstärke
    print(f"\nLautstärke:")
    print(f"  Peak:        {analysis.peak_db:.1f} dB")
    print(f"  RMS:         {analysis.rms_db:.1f} dB")
    print(f"  Stille:      {analysis.silence_ratio*100:.0f}%")

    # Qualitätsscore
    score = analysis.quality_score
    score_bar = "█" * (score // 10) + "░" * (10 - score // 10)
    print(f"\nQualität: [{score_bar}] {score}/100")

    # Issues
    if analysis.issues:
        print(f"\nHinweise:")
        for issue in analysis.issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"\n✅ Keine Probleme erkannt")


def main():
    if not check_ffmpeg():
        print("❌ FFmpeg nicht gefunden. Bitte installieren:")
        print("   brew install ffmpeg  (macOS)")
        print("   apt install ffmpeg   (Ubuntu)")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Sammle alle WAV-Dateien
    if input_path.is_dir():
        wav_files = list(input_path.glob("*.wav"))
    elif input_path.exists():
        wav_files = [input_path]
    else:
        print(f"❌ Pfad nicht gefunden: {input_path}")
        sys.exit(1)

    if not wav_files:
        print(f"❌ Keine WAV-Dateien gefunden in: {input_path}")
        sys.exit(1)

    print(f"\n🎤 Voice Sample Preparation Tool für XTTS v2")
    print(f"   Gefunden: {len(wav_files)} Datei(en)")

    # Analysieren
    analyses = []
    for wav_file in wav_files:
        analysis = analyze_sample(wav_file)
        analyses.append(analysis)
        print_analysis(analysis)

    # Fragen ob optimieren
    needs_optimization = any(a.issues for a in analyses)

    if needs_optimization:
        print(f"\n{'='*60}")
        response = input("\n🔧 Samples automatisch optimieren? [j/N] ").strip().lower()

        if response in ['j', 'ja', 'y', 'yes']:
            print("\nOptimiere Samples...")

            for analysis in analyses:
                if analysis.issues:
                    print(f"\n  Verarbeite: {analysis.path.name}")

                    # Backup erstellen
                    backup_path = analysis.path.with_suffix('.original.wav')
                    if not backup_path.exists():
                        shutil.copy(str(analysis.path), str(backup_path))
                        print(f"    Backup: {backup_path.name}")

                    # Optimieren
                    optimize_sample(analysis.path)
                    print(f"    ✅ Optimiert")

                    # Neu analysieren
                    new_analysis = analyze_sample(analysis.path)
                    print(f"    Neue Qualität: {new_analysis.quality_score}/100")

            print("\n✅ Alle Samples optimiert!")
    else:
        print("\n✅ Alle Samples sind bereits optimal!")

    # Zusammenfassung
    print(f"\n{'='*60}")
    print("Zusammenfassung für XTTS v2:")
    print(f"{'='*60}")

    for analysis in analyses:
        new_analysis = analyze_sample(analysis.path)
        status = "✅" if new_analysis.quality_score >= 70 else "⚠️"
        print(f"  {status} {analysis.path.name}: {new_analysis.quality_score}/100")

    print(f"\nSamples bereit für: samples/host.wav und samples/cohost.wav")


if __name__ == "__main__":
    main()
