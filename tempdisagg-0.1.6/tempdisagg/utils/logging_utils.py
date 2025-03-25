import logging

# Usage: from tempdisagg.utils.logging_utils import VerboseLogger, VerboseFilter

class VerboseFilter(logging.Filter):
    """
    Custom logging filter that allows log messages only if `verbose` is True.
    """
    def __init__(self, verbose):
        # Initialize the parent Filter class
        super().__init__()
        # Store the verbosity setting
        self.verbose = verbose

    def filter(self, record):
        """
        Determine if a log record should be emitted.

        Parameters:
        - record: logging.LogRecord

        Returns:
        - bool: True if verbose is enabled, False otherwise.
        """
        return self.verbose


class VerboseLogger:
    """
    Factory for creating loggers with optional verbosity.
    Automatically disables propagation and applies VerboseFilter.
    """
    def __init__(self, name, verbose=False):
        # Create a named logger
        self.logger = logging.getLogger(name)

        # Set logging level to INFO
        self.logger.setLevel(logging.INFO)

        # Disable log propagation to avoid duplicate messages
        self.logger.propagate = False

        # Only add handler and filters if they haven't been added yet
        if not self.logger.handlers:
            # Create stream handler for console output
            handler = logging.StreamHandler()

            # Set log format (simple message only)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Create filter based on verbosity flag
            verbose_filter = VerboseFilter(verbose)

            # Add filter to both handler and logger to ensure consistency
            handler.addFilter(verbose_filter)
            self.logger.addFilter(verbose_filter)

            # Attach handler to logger
            self.logger.addHandler(handler)

    def get_logger(self):
        """
        Return configured logger instance.

        Returns:
        - logging.Logger: Configured logger with verbosity control.
        """
        return self.logger
