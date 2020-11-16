"""Importing this module initialises prettified/colorful console logging
"""

# External Dependencies:
from colors import color
import logging
from string import Template

# Configuration:
LEVELS = {
    "CRITICAL": {
        "name": "CRIT",
        "fg": "black",
        "bg": "red",
    },
    "ERROR": {
        "name": "ERROR",
        "fg": "red",
    },
    "WARNING": {
        "name": "WARN",
        "fg": "yellow",
    },
    "INFO": {
        "name": "INFO",
        "fg": "green",
    },
    "DEBUG": {
        "name": "DEBUG",
        "fg": "cyan",
    },
    "SILLY": {
        "name": "SILLY",
        "fg": "magenta",
    }
}


class ColorfulFormatter(logging.Formatter):
    def __init__(self, fmt, **kwargs):
        if (kwargs["style"] != "$"):
            # Only Templates support partial substitution, which this formatter requires. Templates use $ style syntax:
            raise ValueError(
                "Only style='$' is supported because this formatter uses String Templates")

        logging.Formatter.__init__(self, fmt, **kwargs)
        tFmt = Template(fmt)
        # This formatter actually works by initialising a pool of different formatters - one for each loglevel.
        self.formatters = {
            level: logging.Formatter(
                tFmt.safe_substitute({"levelname": color(
                    LEVELS[level]["name"],
                    fg=LEVELS[level].get("fg"),
                    bg=LEVELS[level].get("bg")
                )}),
                **kwargs
            )
            for level in LEVELS}

    def format(self, record):
        if (record.levelname in self.formatters):
            return self.formatters[record.levelname].format(record)
        else:
            return super().format(record)


class ColorfulLogger(logging.Logger):
    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)
        console = logging.StreamHandler()
        console.setFormatter(ColorfulFormatter(
            color("[${asctime}]", fg="grey") + " ${levelname} " +
            color("(${name}): ", fg="grey") + "${message}",
            style="$"
        ))
        self.addHandler(console)


logging.setLoggerClass(ColorfulLogger)
