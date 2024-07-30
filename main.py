import subprocess
import uvicorn
from multiprocessing import Process

def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

def run_fastapi():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    p1 = Process(target=run_streamlit)
    p2 = Process(target=run_fastapi)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
