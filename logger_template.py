"""
Robust Python Logger Template

This file provides a reusable logger setup for Python projects. It configures logging to a timestamped file, supports INFO/ERROR/WARNING levels, and can be imported across modules.
"""
import logging
import os
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file with current date
log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Example usage:
if __name__ == "__main__":
    logger.info("Logger initialized and ready.")
    try:
        # Simulate error
        1 / 0
    except Exception as e:
        logger.error("An error occurred", exc_info=True)
    logger.warning("This is a warning message.")
    logger.info("Script finished.")

"""
How to use in other files:
from logger_template import logger
logger.info("Your message here")
logger.error("Error message", exc_info=True)
"""
