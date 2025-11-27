import os
os.system("uvicorn server.api:app --port 8000 &")
os.system("streamlit run ui/ui_app.py")