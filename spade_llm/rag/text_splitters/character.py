"""Character-based text splitters."""

import logging
import re
from typing import List, Optional, Any, Literal

from .base import TextSplitter

logger = logging.getLogger(__name__)


def _split_text_with_regex(
    text: str, 
    separator: str, 
    keep_separator: bool | Literal["start", "end"] = False
) -> List[str]:
    """Split text using regex pattern with optional separator retention.
    
    Args:
        text: Text to split
        separator: Regex pattern to split on
        keep_separator: Whether and where to keep the separator
        
    Returns:
        List of text splits
    """
    if not separator:
        return list(text)
    
    if not keep_separator:
        # Simple split without keeping separator
        splits = re.split(separator, text)
    else:
        # Split while keeping the separator
        # Parentheses in the pattern keep the delimiters in the result
        splits_ = re.split(f"({separator})", text)
        
        if keep_separator == "end":
            # Attach separator to the end of preceding text
            splits = [splits_[i] + splits_[i + 1] 
                     for i in range(0, len(splits_) - 1, 2)]
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = [*splits, splits_[-1]] if splits_ else []
        else:  # "start" or True
            # Attach separator to the start of following text
            splits = [splits_[i] + splits_[i + 1] 
                     for i in range(1, len(splits_), 2)]
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = [splits_[0], *splits] if splits_ else []
    
    # Filter out empty strings
    return [s for s in splits if s]


class CharacterTextSplitter(TextSplitter):
    """Split text based on a single separator character or pattern.
    
    This splitter divides text using a specified separator (which can be a 
    literal string or regex pattern) and then merges the splits into 
    appropriately sized chunks.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the character text splitter.
        
        Args:
            separator: String or regex pattern to split on (default: double newline)
            is_separator_regex: Whether separator is a regex pattern (default: False)
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(**kwargs)
        self.separator = separator
        self.is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Splits text by the separator, then merges into appropriately sized chunks
        with overlap. Handles regex separators and lookaround patterns correctly.
        
        Args:
            text: Text string to split into chunks
            
        Returns:
            List of text chunks respecting size and overlap configuration
        """
        if not text:
            return []
        
        # Prepare separator pattern
        sep_pattern = self.separator if self.is_separator_regex else re.escape(self.separator)
        
        # Initial split (keeping separator if requested)
        splits = _split_text_with_regex(text, sep_pattern, keep_separator=self.keep_separator)
        
        # Detect zero-width lookaround patterns (don't re-insert these)
        lookaround_prefixes = ("(?=", "(?<!", "(?<=", "(?!")
        is_lookaround = self.is_separator_regex and any(
            self.separator.startswith(p) for p in lookaround_prefixes
        )
        
        # Decide merge separator:
        # - If keep_separator or lookaround: don't re-insert separator
        # - Otherwise: re-insert the literal separator
        merge_sep = ""
        if not (self.keep_separator or is_lookaround):
            merge_sep = self.separator
        
        # Merge splits into appropriately sized chunks
        return self._merge_splits(splits, merge_sep)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Split text recursively using multiple separators in order of preference.
    
    This splitter attempts to create more natural text boundaries by trying
    multiple separators in order of preference. If a chunk is still too large
    after splitting with one separator, it recursively tries the next separator
    in the list.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the recursive character text splitter.
        
        Args:
            separators: List of separators to try in order (uses sensible defaults if None)
            keep_separator: Whether and where to keep the separator
            is_separator_regex: Whether separators are regex patterns
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(keep_separator=keep_separator, **kwargs)
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text with the given separators.
        
        Args:
            text: Text string to split recursively
            separators: List of separators to try in order of preference
            
        Returns:
            List of text chunks with natural boundaries preserved
        """
        final_chunks = []
        
        # Find the first separator that exists in the text
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            sep_pattern = sep if self.is_separator_regex else re.escape(sep)
            if not sep:
                separator = sep
                break
            if re.search(sep_pattern, text):
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # Split by the chosen separator
        sep_pattern = separator if self.is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, sep_pattern, keep_separator=self.keep_separator)
        
        # Merge splits, recursively splitting chunks that are too large
        good_splits = []
        merge_sep = "" if self.keep_separator else separator
        
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                # This split is small enough
                good_splits.append(split)
            else:
                # Split is too large, need to handle it
                if good_splits:
                    # First, merge and save the good splits we've accumulated
                    merged_text = self._merge_splits(good_splits, merge_sep)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                if not new_separators:
                    # No more separators to try, just add the large chunk
                    final_chunks.append(split)
                else:
                    # Recursively split with remaining separators
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)
        
        # Don't forget the remaining good splits
        if good_splits:
            merged_text = self._merge_splits(good_splits, merge_sep)
            final_chunks.extend(merged_text)
        
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks obtained after splitting.
        """
        if not text:
            return []
        
        return self._split_text(text, self.separators)