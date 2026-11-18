# PathSnap: running and two-laptop deployment

## Install (both laptops, or just Laptop 1 for the backend)
```powershell
cd PathSnap
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Dataset and artifacts (Laptop 1 only)
- Dataset: place the `Student folder` contents at `data/raw/` (a junction also works), or set `PATHSNAP_DATASET`.
- Trained artifacts live in `artifacts/`: `xgb_tuned.json`, `model_config.json`, `hmm_params.json`, `transition_table.json`.
- To rebuild everything: `01_build_dataset.py` → `02_optuna_xgboost.py` → `03_train_final_models.py` → `04_hmm_decode_and_evaluate.py`.

## Start the backend (Laptop 1)
```powershell
.venv\Scripts\python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Logs go to the console and `logs/api.log`. Each request shows an 8-character request ID, the processing stages and the completion status.

Health check: `http://<LAPTOP1-IP>:8000/api/health`

## Start the frontend (Laptop 2)
```powershell
set PATHSNAP_API_URL=http://<LAPTOP1-IP>:8000
.venv\Scripts\python -m streamlit run frontend\app.py --server.address 0.0.0.0 --server.port 8501
```
Open `http://localhost:8501` on Laptop 2. The backend URL can also be changed in the sidebar.

## Connecting the two laptops
1. Put both laptops on the same Wi-Fi or LAN.
2. Find Laptop 1's address: `ipconfig` → IPv4 address (for example `192.168.1.50`).
3. Allow inbound TCP 8000 on Laptop 1 (Windows Defender Firewall → allow the Python executable on private networks, or add an inbound rule for port 8000).
4. Test from Laptop 2 before starting the UI:
   ```powershell
   curl http://192.168.1.50:8000/api/health
   ```
   It should return `"status": "ok"`. If it times out, the firewall or network is blocking the connection.

## API summary
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model loaded, registered trips |
| GET | `/api/trips` | Evaluation trips (`trip_id`, `partition`) |
| POST | `/api/predict` | Body `{"trip_id": "00000000"}`; returns routes and evaluation metrics |
| GET | `/api/trips/{trip_id}/network` | Candidate road-arc geometry for the map |

Reference routes are read only after predictions are made, and only to compute metrics. They are never included in responses.

## Known gap
The evaluator-testing partition was not in the supplied dataset. The backend lists only the 10 student-test trips until the evaluator folder is provided. Set `PATHSNAP_EVALUATOR_DATASET` to that folder and restart the backend to register those trips.
