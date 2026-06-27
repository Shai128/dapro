import os
import sys
import atexit

ON_WINDOWS = os.name == "nt"
if ON_WINDOWS:
    import msvcrt
else:
    import fcntl

_lifetime_locks = {}  # path -> open file object

def _release_all_locks():
    """Best-effort release of all held locks. Registered with atexit."""
    for path, f in list(_lifetime_locks.items()):
        try:
            if ON_WINDOWS:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                f.close()
            except Exception:
                pass
        finally:
            _lifetime_locks.pop(path, None)

atexit.register(_release_all_locks)

def acquire_lifetime_lock_or_exit(path: str, exit_code: int = 2):
    """
    Acquire an exclusive lock on `path` and hold it for the process lifetime.
    If the lock cannot be acquired (already held), prints an error and exits with exit_code.

    - Creates the file if it doesn't exist.
    - Opens it and attempts an exclusive, non-blocking lock.
    - On success: keeps the open file object in `_lifetime_locks` so the lock is held until process exit.
    - On failure: prints diagnostic and exits immediately.
    """
    path = os.path.abspath(path)

    if path in _lifetime_locks:
        # already locked by this process
        return

    # Ensure parent directory exists (optional; remove if you prefer to fail)
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Open file in binary append+read mode to ensure it exists and allows locking on Windows.
    # We keep the file object open to hold the lock.
    try:
        f = open(path, "a+b")
    except Exception as e:
        print(f"ERROR: cannot open file {path}: {e}", file=sys.stderr)
        sys.exit(exit_code)

    try:
        if ON_WINDOWS:
            # Lock one byte (msvcrt.locking); non-blocking mode: LK_NBLCK
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                # cannot obtain lock
                f.close()
                print(f"ERROR: failed to acquire lock on {path}; another process holds it.", file=sys.stderr)
                sys.exit(exit_code)
        else:
            # POSIX flock exclusive, non-blocking
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                f.close()
                print(f"ERROR: failed to acquire lock on {path}; another process holds it.", file=sys.stderr)
                sys.exit(exit_code)
    except Exception as e:
        # any other unexpected error
        try:
            f.close()
        except Exception:
            pass
        print(f"ERROR: unexpected failure while locking {path}: {e}", file=sys.stderr)
        sys.exit(exit_code)

    # Success: keep file object open for lifetime
    _lifetime_locks[path] = f

def is_locked_by_me(path: str) -> bool:
    """Return True if current process already holds the lifetime lock for path."""
    return os.path.abspath(path) in _lifetime_locks
