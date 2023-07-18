run_server:
	uvicorn main:app --host 127.0.0.1 --port 30006 --reload

run_client:
	python -m streamlit run frontend.py --server.port 30007 --server.fileWatcherType none

run_app: run_server run_client