# scripts/utils.py (Shared utility for logging; modular for all scripts)
import logging
import os
import sys

# Dynamic path fix for config.py in root
bot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(bot_root)
import config

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with file and console handlers. Levels based on config.DEBUG_MODE.
    """
    logger = logging.getLogger(name)
    level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    logger.setLevel(level)
    
    # File handler
    log_dir = os.path.join(config.BOT_NEW_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'bot.log'))
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger