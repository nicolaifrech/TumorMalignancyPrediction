from time import strftime

def print_verbose(msg, verbose=True, level="INFO"):
    if verbose:
        timestamp = strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {msg}")

