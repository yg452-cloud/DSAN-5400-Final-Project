# data/text_cleaner.py

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and normalize Reddit comment text."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove emojis and special unicode characters
        text = re.sub(r'[^\w\s,.!?;:\'\"-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Clean all text in a DataFrame column.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with cleaned text
        """
        logger.info(f"Cleaning {len(df)} texts...")
        
        df = df.copy()
        df['text_clean'] = df[text_column].apply(TextCleaner.clean_text)
        
        # Remove empty texts after cleaning
        original_len = len(df)
        df = df[df['text_clean'].str.len() > 0]
        removed = original_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} empty texts after cleaning")
        
        logger.info(f"Cleaning complete. {len(df)} texts remaining.")
        return df


if __name__ == "__main__":
    test_texts = [
        "Check this out! http://example.com",
        "This is <b>bold</b> text",
        "[Click here](http://test.com) for more",
        "Normal text with punctuation!"
    ]
    
    print("=== Testing Text Cleaner ===")
    for text in test_texts:
        cleaned = TextCleaner.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}\n")