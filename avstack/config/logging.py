import logging

_LOGGER = logging.getLogger("avstack")


def print_log(astr, *args, level=logging.INFO, logger=None, **kwargs):
    """Emit an avstack log message, honoring its level.

    Callers pass a ``level`` (e.g. registry build/import notices use ``logging.DEBUG``; genuine
    problems use ``logging.WARNING``). Previously this ignored ``level`` and ``print()``ed
    everything unconditionally, so routine registry chatter flooded stdout. Now messages route
    through the standard ``logging`` module at their level, so DEBUG/INFO chatter stays quiet by
    default while warnings/errors still surface. Re-enable the chatter with:

        logging.getLogger("avstack").setLevel(logging.DEBUG)
    """
    _LOGGER.log(level if isinstance(level, int) else logging.INFO, astr)
