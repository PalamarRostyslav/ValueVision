"""
Environment setup and authentication utilities.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login
import logging

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment variables and API authentication."""
    
    def __init__(self):
        self.setup_environment()
    
    def setup_environment(self) -> None:
        """Load environment variables and set up API keys."""
        load_dotenv(override=True)
        
        api_keys = {
            'OPENAI_API_KEY': 'default',
            'ANTHROPIC_API_KEY': 'default', 
            'HF_TOKEN': 'default'
        }
        
        for key, default_value in api_keys.items():
            os.environ[key] = os.getenv(key, default_value)
    
    def authenticate_huggingface(self) -> None:
        """Authenticate with Hugging Face using the provided token."""
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token or hf_token == 'default':
            logger.warning("No valid HF_TOKEN found")
            return
            
        try:
            login(hf_token, add_to_git_credential=True)
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            raise
    
    @property
    def hf_token(self) -> str:
        """Get the Hugging Face token."""
        return os.environ.get('HF_TOKEN', 'default')
