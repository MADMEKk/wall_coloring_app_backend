import os
import sys
import socket
import subprocess
import threading
import time

# Get the local IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Start the backend Flask server
def start_backend(ip_address):
    # Activate virtual environment if present
    if os.name == 'nt':  # Windows
        activate_script = os.path.join('venv', 'Scripts', 'activate')
        python_cmd = os.path.join('venv', 'Scripts', 'python.exe')
    else:  # Unix/Linux
        activate_script = os.path.join('venv', 'bin', 'activate')
        python_cmd = os.path.join('venv', 'bin', 'python')
        
    if not os.path.exists(python_cmd):
        # Fall back to system Python
        python_cmd = 'python'
    
    # Start the Flask app
    print(f"Starting backend server at http://{ip_address}:5000")
    backend_process = subprocess.Popen(
        [python_cmd, 'app.py'],
        # Redirect output
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Start a thread to monitor and print the backend output
    def monitor_output():
        for line in backend_process.stdout:
            print(f"[Backend] {line.strip()}")
    
    threading.Thread(target=monitor_output, daemon=True).start()
    
    # Give the backend a moment to start
    time.sleep(2)
    
    return backend_process

# Start a simple HTTP server for the frontend
def start_frontend(ip_address):
    web_dir = os.path.join(os.getcwd(), 'web_html')
    
    # Make sure the web directory exists
    if not os.path.exists(web_dir):
        print(f"Error: Web directory not found at {web_dir}")
        return None
    
    print(f"Starting frontend server to serve files from {web_dir}")
    
    # Use Python's built-in HTTP server, exactly as the user did manually
    frontend_cmd = [sys.executable, '-m', 'http.server', '8000']
    
    # Start the process in the web directory
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=web_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Start a thread to monitor and print the frontend output
    def monitor_output():
        for line in frontend_process.stdout:
            print(f"[Frontend] {line.strip()}")
    
    threading.Thread(target=monitor_output, daemon=True).start()
    
    return frontend_process, 8000

def main():
    try:
        # Get the local IP address
        ip_address = get_local_ip()
        print(f"Detected local IP address: {ip_address}")
        
        # Start the backend server
        backend_process = start_backend(ip_address)
        
        # Start the frontend server as a separate process
        frontend_process, frontend_port = start_frontend(ip_address)
        if not frontend_process:
            print("Failed to start frontend server. Exiting.")
            return
        
        print(f"\n==================================================")
        print(f"App running!")
        print(f"Access options:")
        print(f"   * Local access: http://localhost:{frontend_port}")
        print(f"   * LAN access: http://{ip_address}:{frontend_port}")
        print(f"     (Share this URL with other devices on your network)")
        print(f"\nBackend API:")
        print(f"   * http://{ip_address}:5000")
        print(f"\nIMPORTANT: You may need to manually set the backend URL")
        print(f"in the settings page of the app after loading it.")
        print(f"==================================================\n")
        
        # Keep the main process running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("Backend process has stopped. Exiting.")
                break
                
            if frontend_process.poll() is not None:
                print("Frontend process has stopped. Exiting.")
                break
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Try to clean up processes
        try:
            if 'backend_process' in locals() and backend_process.poll() is None:
                backend_process.terminate()
            if 'frontend_process' in locals() and frontend_process.poll() is None:
                frontend_process.terminate()
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()