"""
Generiert natürliche Podcast-Dialoge aus Präsentationsinhalten.
Erweitert mit Kapitel-Tracking und Struktur-Ansagen für Accessibility.
"""
import ollama
from dataclasses import dataclass, field
from typing import Generator, Optional
import json
import re
import logging

from .config import settings
from .extractor import PresentationContent
from .chapter_manager import ChapterMarker

logger = logging.getLogger(__name__)


@dataclass
class SlideContext:
    """Kontext einer Folie für Orientierung."""
    slide_number: int
    total_slides: int
    title: Optional[str] = None
    section_name: Optional[str] = None
    is_section_start: bool = False


@dataclass
class DialogLine:
    """Eine einzelne Zeile im Dialog."""
    speaker: str  # "host" oder "cohost"
    text: str
    emotion: str = "neutral"
    audio_cue: Optional[str] = None  # "slide_transition", "section_start", etc.
    slide_context: Optional[SlideContext] = None
    marks_chapter_start: bool = False


@dataclass
class PodcastScript:
    """Komplettes Podcast-Skript mit Kapitelinformationen."""
    title: str
    lines: list[DialogLine] = field(default_factory=list)
    chapters: list[ChapterMarker] = field(default_factory=list)

    def to_ssml_segments(self) -> list[dict]:
        """Konvertiert zu Segmenten für TTS."""
        return [
            {
                "speaker": line.speaker,
                "text": line.text,
                "emotion": line.emotion,
                "audio_cue": line.audio_cue
            }
            for line in self.lines
        ]

    def get_audio_cue_map(self) -> dict[int, str]:
        """Erstellt Mapping von Zeilen-Index zu Audio-Cue."""
        return {
            i: line.audio_cue
            for i, line in enumerate(self.lines)
            if line.audio_cue
        }


DIALOG_SYSTEM_PROMPT = """Du bist ein Skript-Autor für Bildungspodcasts, die besonders für Menschen mit
Sehbehinderung geeignet sein sollen. Du schreibst natürliche, engagierte Dialoge zwischen zwei
Podcast-Hosts, die gemeinsam eine Präsentation besprechen.

Die Hosts:
- {host_name}: Der Hauptmoderator. Führt durch die Inhalte, erklärt Konzepte, fasst zusammen.
- {cohost_name}: Der Co-Host. Stellt Rückfragen, bringt Beispiele, macht komplexe Dinge greifbar.

WICHTIGE REGELN:
1. Schreibe NATÜRLICHE Sprache - so wie Menschen wirklich reden
2. Keine steifen Formulierungen wie "Auf dieser Folie sehen wir..."
3. Vermeide Wiederholungen und redundante Einleitungen
4. Die Hosts unterbrechen sich gelegentlich, reagieren aufeinander
5. Nutze Füllwörter sparsam aber natürlich: "also", "quasi", "sozusagen"
6. Bei komplexen Themen: erst vereinfachen, dann Details
7. Humor ist erlaubt, aber dezent
8. KEINE Emojis oder Sonderzeichen
9. Jede Sprechzeile sollte 1-3 Sätze lang sein

ACCESSIBILITY-REGELN:
10. Beschreibe Bilder und Diagramme verbal - die Zuhörer können sie nicht sehen
11. Bei Diagrammen: Beschreibe die Kernaussage, nicht jedes Detail
12. Gib gelegentlich Orientierung: "Wir sind jetzt etwa in der Mitte" oder ähnlich
13. Bei Abschnittswechseln: Kurz zusammenfassen was besprochen wurde

OUTPUT-FORMAT (strikt einhalten!):
Gib NUR valides JSON zurück, keine Erklärungen davor oder danach.
Das JSON ist ein Array von Objekten:

[
  {{"speaker": "host", "text": "...", "slide": 1, "section_start": false}},
  {{"speaker": "cohost", "text": "...", "slide": 1}},
  ...
]

Das "slide"-Feld gibt an, zu welcher Folie der Text gehört.
Das "section_start"-Feld (optional) markiert den Beginn eines neuen Abschnitts.

SPRACHSTIL-BEISPIELE:

SCHLECHT (zu formal):
"Willkommen zu unserem Podcast. Heute besprechen wir die Präsentation zum Thema X."

GUT (natürlich):
"Hey, heute hab ich was Spannendes mitgebracht - eine Präsentation über X, und ich finde,
da sind ein paar echt interessante Punkte drin."

SCHLECHT (nicht barrierefrei):
"Wie ihr auf dem Diagramm sehen könnt..."

GUT (barrierefrei):
"Da gibt's ein Diagramm, das zeigt wie sich die Zahlen entwickelt haben - kurz gesagt:
es geht steil nach oben, von 10 auf über 50 in nur drei Jahren."
"""

DIALOG_USER_PROMPT = """Hier ist der Inhalt einer Präsentation mit {total_slides} Folien.
Erstelle daraus einen natürlichen Podcast-Dialog zwischen {host_name} und {cohost_name}.

Der Podcast sollte:
- Mit einer kurzen, lockeren Begrüßung starten (NICHT "Willkommen zu unserem Podcast")
- Die wichtigsten Punkte der Präsentation verständlich erklären
- Komplexe Inhalte durch Beispiele oder Analogien greifbar machen
- Bildbeschreibungen natürlich einbauen (z.B. "Da gibt's eine Grafik die zeigt...")
- Gelegentlich Orientierung geben ("Wir sind jetzt bei Folie 3 von 12")
- Mit einem kurzen Fazit enden

PRÄSENTATIONSINHALT:
{content}

Generiere jetzt den Dialog als JSON-Array. Vergiss nicht das "slide"-Feld bei jeder Zeile:"""


def clean_json_response(response: str) -> str:
    """Bereinigt LLM-Output und extrahiert JSON."""
    # Entferne Markdown Code-Blocks falls vorhanden
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Finde JSON-Array
    match = re.search(r'\[[\s\S]*\]', response)
    if match:
        return match.group(0)

    return response.strip()


def _create_orientation_text(slide_number: int, total_slides: int) -> str:
    """Erstellt Orientierungstext basierend auf Position."""
    progress = slide_number / total_slides

    if slide_number == 1:
        return ""  # Erste Folie braucht keine Orientierung
    elif progress < 0.3:
        return f"Wir sind noch am Anfang, bei Folie {slide_number} von {total_slides}. "
    elif progress < 0.5:
        return f"Wir haben etwa ein Drittel geschafft. "
    elif progress < 0.7:
        return f"Wir sind jetzt in der Mitte der Präsentation. "
    elif progress < 0.9:
        return f"Wir kommen langsam zum Ende, Folie {slide_number} von {total_slides}. "
    else:
        return f"Wir sind fast durch, noch {total_slides - slide_number + 1} Folien. "


def _chunk_slides(slides: list, max_slides_per_chunk: int = 10) -> list[list]:
    """Teilt Folien in Chunks auf."""
    chunks = []
    for i in range(0, len(slides), max_slides_per_chunk):
        chunks.append(slides[i:i + max_slides_per_chunk])
    return chunks


def _generate_chunk_dialog(
    chunk_slides: list,
    chunk_index: int,
    total_chunks: int,
    total_slides: int,
    title: str
) -> list[dict]:
    """Generiert Dialog für einen Chunk von Folien."""

    # Chunk-spezifischen Prompt erstellen
    chunk_text = []
    for slide in chunk_slides:
        chunk_text.append(f"### Folie {slide.slide_number}")
        chunk_text.append(slide.to_text())

    chunk_content = "\n".join(chunk_text)

    # Kontext-Info für das Modell
    if chunk_index == 0:
        context = "Dies ist der ANFANG der Präsentation. Starte mit einer lockeren Begrüßung."
    elif chunk_index == total_chunks - 1:
        context = "Dies ist das ENDE der Präsentation. Schließe mit einem Fazit ab."
    else:
        context = f"Dies ist Teil {chunk_index + 1} von {total_chunks} der Präsentation. Führe den Dialog natürlich fort."

    system_prompt = f"""Du bist ein Podcast-Skript-Autor. Schreibe natürliche Dialoge zwischen {settings.host_name} (Host) und {settings.cohost_name} (Co-Host).

REGELN:
- Natürliche Sprache, keine steifen Formulierungen
- Beschreibe Bilder/Diagramme verbal für blinde Zuhörer
- Jede Zeile 1-3 Sätze
- KEINE Emojis

OUTPUT: NUR valides JSON, keine Erklärungen:
[{{"speaker": "host", "text": "...", "slide": 1}}, {{"speaker": "cohost", "text": "...", "slide": 1}}, ...]"""

    user_prompt = f"""{context}

Präsentation: {title}
Folien {chunk_slides[0].slide_number} bis {chunk_slides[-1].slide_number} von {total_slides}:

{chunk_content}

Generiere 8-15 Dialog-Zeilen als JSON:"""

    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.7,
            "num_predict": 3000,
        }
    )

    raw_response = response['message']['content']
    cleaned_json = clean_json_response(raw_response)

    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON-Parse-Fehler in Chunk {chunk_index}: {e}")
        return []


def generate_dialog(content: PresentationContent) -> PodcastScript:
    """
    Generiert einen Podcast-Dialog aus Präsentationsinhalten.
    Bei langen Präsentationen wird in Chunks verarbeitet.

    Args:
        content: Extrahierte Präsentationsinhalte (mit Bildbeschreibungen)

    Returns:
        PodcastScript mit allen Dialog-Zeilen und Kapitelmarkern
    """
    logger.info(f"Generiere Dialog mit {settings.ollama_model}...")

    # Bei langen Präsentationen: Chunk-Verarbeitung
    if len(content.slides) > 12:
        logger.info(f"Lange Präsentation ({len(content.slides)} Folien) - verarbeite in Chunks")
        chunks = _chunk_slides(content.slides, max_slides_per_chunk=8)

        all_dialog_data = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Verarbeite Chunk {i+1}/{len(chunks)} (Folien {chunk[0].slide_number}-{chunk[-1].slide_number})")
            chunk_dialog = _generate_chunk_dialog(
                chunk_slides=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
                total_slides=content.total_slides,
                title=content.title or "Präsentation"
            )
            all_dialog_data.extend(chunk_dialog)

        dialog_data = all_dialog_data
    else:
        # Kurze Präsentation: Standard-Verarbeitung
        system_prompt = DIALOG_SYSTEM_PROMPT.format(
            host_name=settings.host_name,
            cohost_name=settings.cohost_name
        )

        user_prompt = DIALOG_USER_PROMPT.format(
            host_name=settings.host_name,
            cohost_name=settings.cohost_name,
            total_slides=content.total_slides,
            content=content.to_prompt_text()
        )

        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 8192,
            }
        )

        raw_response = response['message']['content']
        cleaned_json = clean_json_response(raw_response)

        try:
            dialog_data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON-Parse-Fehler: {e}")
            raise ValueError(f"LLM hat kein valides JSON generiert: {e}")

    # In DialogLines konvertieren mit Kapitel-Tracking
    lines = []
    chapters = []
    current_slide = 0
    current_section = None
    slide_titles = {s.slide_number: s.title for s in content.slides}

    for item in dialog_data:
        speaker = item.get("speaker", "host")
        text = item.get("text", "")
        slide = item.get("slide", current_slide)
        section_start = item.get("section_start", False)

        if not text.strip():
            continue

        # Audio-Cue und Kapitel-Markierung bestimmen
        audio_cue = None
        marks_chapter = False

        # Neuer Abschnitt?
        if section_start:
            audio_cue = "section_start"
            marks_chapter = True
            # Kapitel hinzufügen (Zeit wird später beim TTS gefüllt)
            section_name = item.get("section_name", f"Abschnitt ab Folie {slide}")
            chapters.append(ChapterMarker(
                title=section_name,
                start_ms=0,  # Wird später aktualisiert
                slide_number=slide,
                section_name=section_name
            ))
            current_section = section_name

        # Neue Folie?
        elif slide != current_slide and settings.chapter_per_slide:
            audio_cue = "slide_transition"
            marks_chapter = True
            # Kapitel für Folie
            slide_title = slide_titles.get(slide, f"Folie {slide}")
            chapter_title = f"Folie {slide}: {slide_title}" if slide_title else f"Folie {slide}"
            chapters.append(ChapterMarker(
                title=chapter_title,
                start_ms=0,
                slide_number=slide
            ))

        current_slide = slide

        # Slide-Kontext erstellen
        slide_context = SlideContext(
            slide_number=slide,
            total_slides=content.total_slides,
            title=slide_titles.get(slide),
            section_name=current_section,
            is_section_start=section_start
        )

        lines.append(DialogLine(
            speaker=speaker,
            text=text.strip(),
            emotion=item.get("emotion", "neutral"),
            audio_cue=audio_cue,
            slide_context=slide_context,
            marks_chapter_start=marks_chapter
        ))

    # Einleitung und Zusammenfassung als Kapitel hinzufügen
    if chapters:
        # Einleitung am Anfang
        chapters.insert(0, ChapterMarker(
            title="Einleitung",
            start_ms=0
        ))
        # Zusammenfassung am Ende (wenn letzte Zeilen keine Folie haben)
        if lines and (lines[-1].slide_context is None or
                      lines[-1].slide_context.slide_number == content.total_slides):
            chapters.append(ChapterMarker(
                title="Zusammenfassung",
                start_ms=0
            ))

    script = PodcastScript(
        title=content.title or "Podcast",
        lines=lines,
        chapters=chapters
    )

    logger.info(f"Dialog generiert: {len(lines)} Zeilen, {len(chapters)} Kapitel")

    return script


def generate_dialog_streaming(content: PresentationContent) -> Generator[str, None, None]:
    """
    Streaming-Variante für Live-Feedback.
    Yielded Partial-JSON für Progress-Anzeige.
    """
    system_prompt = DIALOG_SYSTEM_PROMPT.format(
        host_name=settings.host_name,
        cohost_name=settings.cohost_name
    )

    user_prompt = DIALOG_USER_PROMPT.format(
        host_name=settings.host_name,
        cohost_name=settings.cohost_name,
        total_slides=content.total_slides,
        content=content.to_prompt_text()
    )

    stream = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        options={
            "temperature": 0.7,
            "num_predict": 8192,
        }
    )

    for chunk in stream:
        yield chunk['message']['content']


# CLI für Tests
if __name__ == "__main__":
    from .extractor import extract_pptx
    from pathlib import Path
    import sys

    if len(sys.argv) > 1:
        content = extract_pptx(Path(sys.argv[1]))
        script = generate_dialog(content)

        print(f"\n{'=' * 60}")
        print(f"PODCAST: {script.title}")
        print(f"{'=' * 60}")

        print(f"\nKapitel ({len(script.chapters)}):")
        for ch in script.chapters:
            print(f"  - {ch.title}")

        print(f"\n{'=' * 60}\n")

        for line in script.lines:
            name = settings.host_name if line.speaker == "host" else settings.cohost_name
            cue = f" [{line.audio_cue}]" if line.audio_cue else ""
            print(f"[{name}]{cue}: {line.text}\n")
