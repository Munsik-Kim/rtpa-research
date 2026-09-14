"""Local POSIX read boundary: never follow a file or intermediate symlink.

Source builds, restricted PT loading and maintenance R4 verification require
POSIX descriptor-relative no-follow file access. The tested environment is
Linux; native Windows support for these paths is not provided by this revision.
Missing capabilities fail closed. Other installed CPU commands have separate
requirements; their Windows compatibility has not been established here.
This is not a general filesystem sandbox.
"""
from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import stat


@contextmanager
def regular_reader(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or ':' in relative or PurePosixPath(relative).is_absolute()
            or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise ValueError('UNSAFE_RELATIVE_READ_PATH')
    if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
        raise ValueError('NOFOLLOW_READ_PLATFORM_UNSUPPORTED')
    descriptors = []
    try:
        fd = os.open(Path(root).resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        parts = relative.split('/')
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        descriptors.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('NONREGULAR_READ_REJECTED')
        # fdopen owns only a duplicate; cleanup retains every original descriptor.
        with os.fdopen(os.dup(fd), 'rb') as stream:
            yield stream
    except OSError as exc:
        raise ValueError('NOFOLLOW_PATH_READ_REJECTED:' + relative) from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def regular_bytes(root, relative):
    with regular_reader(root, relative) as stream:
        before = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
        if (before.st_size != len(data) or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns):
            raise ValueError('SOURCE_CHANGED_DURING_READ')
        return data
