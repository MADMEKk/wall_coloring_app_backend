# Wall Color Changer - Backend and Frontend

This project allows you to change colors and add textures to walls and other surfaces in interior design photos, using AI segmentation.

## Quick Start

### Windows Users
Simply double-click the `start_app.bat` file to run both the backend and frontend together.

### All Users
Run the following command to start both the backend and frontend:
```bash
python start_app.py
```

The application will:
1. Start the backend server with AI processing capabilities
2. Configure the frontend to connect to the backend automatically
3. Open a web server for the frontend interface
4. Display the URLs to access the application

Once started, open your browser and navigate to:
- On your computer: http://localhost:8000
- From other devices on your network: http://YOUR_IP_ADDRESS:8000 (where YOUR_IP_ADDRESS will be shown in the console)

## Manual Setup Instructions

If you prefer to set up and run the components separately, follow these steps:

### Backend Setup

1. **Prerequisites**
   - Python 3.8 or newer
   - pip (Python package installer)

2. **Create a Virtual Environment**
   ```bash
   # Create the virtual environment
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - **On Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **On macOS and Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Backend Server**
   ```bash
   python app.py
   ```
   The server will start on http://0.0.0.0:5000

### Frontend Access

The frontend is pre-built using Flutter. To serve it separately:

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Start a simple HTTP server:
   - **Python 3:**
     ```bash
     python -m http.server 8000
     ```
   - **Python 2:**
     ```bash
     python -m SimpleHTTPServer 8000
     ```

3. Open your browser and go to http://localhost:8000

## Project Structure
- `app.py` - Backend Flask server
- `texture_generator.py` - Texture generation utilities
- `textures/` - Texture image files
- `web/` - Pre-built Flutter web app
- `start_app.py` - Integration script to run both frontend and backend