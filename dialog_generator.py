"""
Generiert natürliche Podcast-Dialoge aus Präsentationsinhalten.
Das ist das Herzstück des Systems.
"""
import ollama
from dataclasses import dataclass
from typing import Generator
import json
import re

from .config import settings
from .extractor import PresentationContent


@dataclass
class DialogLine:
    """Eine einzelne Zeile im Dialog."""
    speaker: str  # "host" oder "cohost"
    text: str
    emotion: str = "neutral"  # für spätere TTS-Steuerung


@dataclass
class PodcastScript:
    """Komplettes Podcast-Skript."""
    title: str
    lines: list[DialogLine]
    
    def to_ssml_segments(self) -> list[dict]:
        """Konvertiert zu Segmenten für TTS."""
        return [
            {
                "speaker": line.speaker,
                "text": line.text,
                "emotion": line.emotion
            }
            for line in self.lines
        ]


DIALOG_SYSTEM_PROMPT = """Du bist ein Skript-Autor für Bildungspodcasts. Du schreibst natürliche, 
engagierte Dialoge zwischen zwei Podcast-Hosts, die gemeinsam eine Präsentation besprechen.

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

OUTPUT-FORMAT (strikt einhalten!):
Gib NUR valides JSON zurück, keine Erklärungen davor oder danach.
Das JSON ist ein Array von Objekten:

[
  {{"speaker": "host", "text": "..."}},
  {{"speaker": "cohost", "text": "..."}},
  ...
]

SPRACHSTIL-BEISPIELE:

SCHLECHT (zu formal):
"Willkommen zu unserem Podcast. Heute besprechen wir die Präsentation zum Thema X."

GUT (natürlich):
"Hey, heute hab ich was Spannendes mitgebracht - eine Präsentation über X, und ich finde, 
da sind ein paar echt interessante Punkte drin."

SCHLECHT (roboterhaft):
"Auf Folie 3 wird das Thema Y behandelt. Die Kernaussage ist Z."

GUT (gesprächig):
"Jetzt wirds spannend - die haben sich nämlich angeschaut, wie Y funktioniert. 
Und was dabei rauskam, hat mich ehrlich gesagt überrascht."
"""

DIALOG_USER_PROMPT = """Hier ist der Inhalt einer Präsentation. Erstelle daraus einen 
natürlichen Podcast-Dialog zwischen {host_name} und {cohost_name}.

Der Podcast sollte:
- Mit einer kurzen, lockeren Begrüßung starten (NICHT "Willkommen zu unserem Podcast")
- Die wichtigsten Punkte der Präsentation verständlich erklären
- Komplexe Inhalte durch Beispiele oder Analogien greifbar machen
- Mit einem kurzen Fazit enden

PRÄSENTATIONSINHALT:
{content}

Generiere jetzt den Dialog als JSON-Array:"""


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


def generate_dialog(content: PresentationContent) -> PodcastScript:
    """
    Generiert einen Podcast-Dialog aus Präsentationsinhalten.
    
    Args:
        content: Extrahierte Präsentationsinhalte
        
    Returns:
        PodcastScript mit allen Dialog-Zeilen
    """
    # Prompts vorbereiten
    system_prompt = DIALOG_SYSTEM_PROMPT.format(
        host_name=settings.host_name,
        cohost_name=settings.cohost_name
    )
    
    user_prompt = DIALOG_USER_PROMPT.format(
        host_name=settings.host_name,
        cohost_name=settings.cohost_name,
        content=content.to_prompt_text()
    )
    
    # Ollama aufrufen
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.7,
            "num_predict": 4096,
        }
    )
    
    # Response parsen
    raw_response = response['message']['content']
    cleaned_json = clean_json_response(raw_response)
    
    try:
        dialog_data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM hat kein valides JSON generiert: {e}\n\nRaw: {raw_response}")
    
    # In DialogLines konvertieren
    lines = []
    for item in dialog_data:
        speaker = item.get("speaker", "host")
        text = item.get("text", "")
        emotion = item.get("emotion", "neutral")
        
        if text.strip():
            lines.append(DialogLine(
                speaker=speaker,
                text=text.strip(),
                emotion=emotion
            ))
    
    return PodcastScript(
        title=content.title or "Podcast",
        lines=lines
    )


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
            "num_predict": 4096,
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
        
        print(f"\n{'='*60}")
        print(f"PODCAST: {script.title}")
        print(f"{'='*60}\n")
        
        for line in script.lines:
            name = settings.host_name if line.speaker == "host" else settings.cohost_name
            print(f"[{name}]: {line.text}\n")
