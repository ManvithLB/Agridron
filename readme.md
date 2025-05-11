# Agridron 🚁

Agridron is a project designed to simplify and enhance agricultural operations using advanced technologies.

## Description

This application leverages modern tools to provide efficient solutions for agricultural management. It is built using Flask and requires a virtual environment for dependency management.

## Steps to Run

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env-flask` file** in the main directory and configure the necessary environment variables. Example:

   ```
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your_secret_key
   ```

5. **Run the Flask application**:

   ```bash
   flask run
   ```

6. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

Enjoy using Agridron! 🚀
