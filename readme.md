# Agridron 🚁

Agridron is a project designed to simplify and enhance agricultural operations using advanced technologies.

## Description

This application leverages modern tools to provide efficient solutions for agricultural management. It is built using Flask and requires a virtual environment for dependency management.

## Steps to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ManvithLB/Agridron
   cd agridron
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
   PORT=5001
   DEBUG=False
   SECRET_KEY=secret_key
   OPENAI_API_KEY=open_api_key
   API_KEY=api_key

   ```

5. **Run the Flask application**:

   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

Enjoy using Agridron! 🚀
