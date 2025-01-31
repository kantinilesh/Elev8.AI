import re
from typing import Dict, Optional, List
from youtube_transcript_api import YouTubeTranscriptApi

class BrailleConverter:
    BRAILLE_MAP = {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 
        'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
        'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 
        'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
        'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 
        'z': '⠵',
        ' ': '⠀', '.': '⠲', ',': '⠂', '!': '⠖',
        # Add more characters as needed
    }

    @classmethod
    def to_braille(cls, text: str) -> str:
        braille_text = [cls.BRAILLE_MAP.get(char.lower(), '⠀') for char in text]
        return ''.join(braille_text)

class YouTubeBrailleTranslator:
    def get_video_transcript(self, video_id: str) -> Optional[List[Dict]]:
        try:
            return YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return None

    def extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r'(?:https?:\/\/)?youtu\.be\/([^&\s]+)',
            r'(?:https?:\/\/)?youtube\.com\/watch\?v=([^&\s]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def translate_transcript_to_braille(self, video_id: str) -> Dict[str, str]:
        transcript = self.get_video_transcript(video_id)
        if not transcript:
            return {
                "original_transcript": "Could not retrieve transcript.",
                "braille_transcript": "⠉⠕⠥⠇⠙⠀⠝⠕⠞⠀⠗⠑⠞⠗⠊⠑⠧⠑⠀⠞⠗⠁⠝⠎⠉⠗⠊⠏⠞⠲"
            }
        return {
            "original_transcript": "\n".join(e['text'] for e in transcript),
            "braille_transcript": "\n".join(BrailleConverter.to_braille(e['text']) for e in transcript),
        }