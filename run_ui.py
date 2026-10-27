import os
import sys
import socket
import webbrowser
import uvicorn

def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def get_available_port(preferred: int = 8000) -> int:
    port = preferred
    while port < preferred + 50:
        if is_port_free(port):
            return port
        port += 1
    return preferred

if __name__ == "__main__":
    port = get_available_port(8000)
    url = f"http://127.0.0.1:{port}"
    
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\n" + "=" * 65)
    print("  [*] Starting DeepDive LangGraph State Studio")
    print(f"  [+] Dashboard URL: {url}")
    print("=" * 65 + "\n")
    
    # Open browser automatically after a short delay
    try:
        webbrowser.open(url)
    except Exception:
        pass
    
    uvicorn.run("backend.server:app", host="127.0.0.1", port=port, reload=False)
