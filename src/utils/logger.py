import logging
import os
import sys
from typing import Optional, Dict


class HCDLogger:
    """
    A logger class for the HCD framework, wrapping the standard logging module.
    Follows a singleton-like pattern for named loggers.
    """
    _loggers: Dict[str, 'HCDLogger'] = {}

    def __init__(self, name: str, log_file: Optional[str] = None, log_level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Only add handlers if they don't exist
        if not self.logger.handlers:
            self._setup_handlers(log_file)
        
        HCDLogger._loggers[name] = self

    def _setup_handlers(self, log_file: Optional[str]):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File Handler
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    @classmethod
    def get_instance(cls, name: str = "hcd", **kwargs) -> 'HCDLogger':
        """Get or create a logger instance."""
        if name not in cls._loggers:
            cls(name, **kwargs)
        return cls._loggers[name]

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str = "hcd", log_file: Optional[str] = None, log_level: int = logging.INFO) -> HCDLogger:
    """Utility function to get a logger instance."""
    return HCDLogger.get_instance(name, log_file=log_file, log_level=log_level)