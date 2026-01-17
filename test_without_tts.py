#!/usr/bin/env python3
"""
Test-Skript zum Testen der Pipeline ohne TTS.
Testet: PPTX-Extraktion, Bildbeschreibung, Dialog-Generierung.
"""
import sys
from pathlib import Path

# App-Modul zum Path hinzufügen
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.extractor import extract_pptx
from app.image_describer import describe_images_in_content
from app.dialog_generator import generate_dialog


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_without_tts.py <pptx_file>")
        print("\nDieses Skript testet:")
        print("  1. PPTX-Extraktion (Text + Bilder)")
        print("  2. KI-Bildbeschreibung mit LLaVA")
        print("  3. Dialog-Generierung mit Ollama")
        return

    pptx_path = Path(sys.argv[1])
    if not pptx_path.exists():
        print(f"Datei nicht gefunden: {pptx_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"PPTX-to-Podcast Test (ohne TTS)")
    print(f"{'=' * 60}")
    print(f"Datei: {pptx_path.name}")
    print(f"Modelle: {settings.ollama_model}, {settings.vision_model}")
    print(f"{'=' * 60}\n")

    # 1. Extraktion
    print("[1/3] Extrahiere PPTX...")
    content = extract_pptx(pptx_path)
    print(f"      → {content.total_slides} Folien")
    total_images = sum(len(s.images) for s in content.slides)
    print(f"      → {total_images} Bilder gefunden")

    for slide in content.slides:
        print(f"\n      Folie {slide.slide_number}: {slide.title or '(kein Titel)'}")
        if slide.images:
            print(f"        - {len(slide.images)} Bild(er)")
        if slide.body_text:
            preview = slide.body_text[0][:50] + "..." if len(slide.body_text[0]) > 50 else slide.body_text[0]
            print(f"        - Text: {preview}")

    # 2. Bildbeschreibung
    if total_images > 0 and settings.image_description_enabled:
        print(f"\n[2/3] Beschreibe {total_images} Bilder mit LLaVA...")
        content = describe_images_in_content(content)

        described = sum(1 for s in content.slides for img in s.images if img.description)
        print(f"      → {described} Bilder beschrieben")

        for slide in content.slides:
            for img in slide.images:
                if img.description:
                    print(f"\n      Folie {slide.slide_number} Bild:")
                    print(f"        \"{img.description[:100]}...\"" if len(img.description) > 100 else f"        \"{img.description}\"")
    else:
        print("\n[2/3] Keine Bilder zu beschreiben (übersprungen)")

    # 3. Dialog-Generierung
    print(f"\n[3/3] Generiere Podcast-Dialog...")
    script = generate_dialog(content)
    print(f"      → {len(script.lines)} Dialog-Zeilen")
    print(f"      → {len(script.chapters)} Kapitel")

    print(f"\n{'=' * 60}")
    print(f"PODCAST: {script.title}")
    print(f"{'=' * 60}")

    print("\nKapitel:")
    for ch in script.chapters:
        print(f"  - {ch.title}")

    print(f"\n{'=' * 60}")
    print("DIALOG-VORSCHAU (erste 10 Zeilen):")
    print(f"{'=' * 60}\n")

    for line in script.lines[:10]:
        name = settings.host_name if line.speaker == "host" else settings.cohost_name
        cue = f" [{line.audio_cue}]" if line.audio_cue else ""
        print(f"[{name}]{cue}:")
        print(f"  {line.text}\n")

    if len(script.lines) > 10:
        print(f"... und {len(script.lines) - 10} weitere Zeilen")

    print(f"\n{'=' * 60}")
    print("Test abgeschlossen!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
