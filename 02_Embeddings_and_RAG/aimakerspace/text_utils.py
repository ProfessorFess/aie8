import os
import re
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import requests


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class YouTubeLoader:
    def __init__(self, url: str, include_metadata: bool = True):
        """
        Initialize YouTubeLoader with a YouTube URL.
        
        :param url: YouTube video URL (supports various formats)
        :param include_metadata: Whether to include video metadata in the document
        """
        self.url = url
        self.include_metadata = include_metadata
        self.documents = []
        self.metadata = {}
        self.video_id = self._extract_video_id(url)
        
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _get_video_metadata(self) -> Dict[str, Any]:
        """Get basic video metadata using YouTube's oEmbed API."""
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={self.url}&format=json"
            response = requests.get(oembed_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Could not fetch video metadata: {e}")
            return {"title": "Unknown", "author_name": "Unknown", "provider_name": "YouTube"}
    
    def _get_transcript(self) -> str:
        """Get video transcript using youtube-transcript-api."""
        try:
            # Create an instance of YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            
            # Try to get transcript in English first
            try:
                transcript_data = api.fetch(self.video_id, languages=['en'])
            except:
                # If English not available, get any available transcript
                transcript_data = api.fetch(self.video_id)
            
            # Combine all transcript segments into a single text
            full_transcript = " ".join([segment.text for segment in transcript_data.snippets])
            return full_transcript
            
        except Exception as e:
            raise ValueError(f"Could not retrieve transcript for video {self.video_id}: {e}")
    
    def load(self):
        """Load YouTube video transcript and metadata."""
        try:
            # Get transcript
            transcript = self._get_transcript()
            
            # Get metadata if requested
            if self.include_metadata:
                self.metadata = self._get_video_metadata()
                
                # Create a formatted document with metadata and transcript
                document = f"""Title: {self.metadata.get('title', 'Unknown')}
Author: {self.metadata.get('author_name', 'Unknown')}
Video ID: {self.video_id}
URL: {self.url}

Transcript:
{transcript}"""
            else:
                document = transcript
            
            self.documents.append(document)
            
        except Exception as e:
            raise ValueError(f"Failed to load YouTube video: {e}")
    
    def load_documents(self) -> List[str]:
        """Load documents and return them."""
        self.load()
        return self.documents
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get video metadata."""
        return self.metadata


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
