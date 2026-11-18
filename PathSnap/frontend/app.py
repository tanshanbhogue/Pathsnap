"""PathSnap evaluator interface (runs on Laptop 2).

Run:  streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
Set the backend address with the environment variable PATHSNAP_API_URL, for example
  PATHSNAP_API_URL=http://192.168.1.50:8000
(or change it in the sidebar). Every number shown comes from the backend response.
"""
from __future__ import annotations

import os

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_S = 180

st.set_page_config(page_title="PathSnap", layout="wide")


# ----------------------------------------------------------------- API helpers
def api_get(base: str, path: str):
    r = requests.get(f"{base}{path}", timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def api_post(base: str, path: str, payload: dict):
    r = requests.post(f"{base}{path}", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def explain_error(exc: Exception, base: str) -> str:
    if isinstance(exc, requests.ConnectionError):
        return (f"Cannot reach the PathSnap backend at {base}. Check that Laptop 1 is running the API, "
                f"that both laptops are on the same network, and that the address is correct.")
    if isinstance(exc, requests.Timeout):
        return "The backend did not respond in time. The request may still be running on Laptop 1."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail") or exc.response.json().get("detail")
        except ValueError:
            detail = exc.response.text[:300]
        return f"Backend returned HTTP {exc.response.status_code}: {detail}"
    return f"Unexpected error: {exc}"


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("Backend URL", value=DEFAULT_API).rstrip("/")
    try:
        health = api_get(api_url, "/api/health")
        if health.get("model_loaded"):
            st.success("Backend online, model loaded")
        else:
            st.warning(f"Backend online but model not loaded: {health.get('model_error')}")
        st.caption(f"Partitions: {', '.join(health.get('partitions', []))}")
        if not health.get("evaluator_partition_configured"):
            st.info("Evaluator partition is not configured on the backend (PATHSNAP_EVALUATOR_DATASET).")
        backend_ok = True
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(explain_error(exc, api_url))
        backend_ok = False

# ----------------------------------------------------------------- main
st.title("PathSnap: route reconstruction")
st.write("Select an evaluation trip. The backend matches each GPS observation to road segments with "
         "XGBoost, then reconstructs the most plausible route with an HMM and Viterbi decoding.")

if not backend_ok:
    st.stop()

try:
    trips_payload = api_get(api_url, "/api/trips")
    trips = trips_payload["trips"]
except Exception as exc:  # noqa: BLE001
    st.error(explain_error(exc, api_url))
    st.stop()

if not trips:
    st.warning("The backend has no evaluation trips registered.")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
cols = st.columns(5)
for i, trip in enumerate(trips):
    with cols[i % 5]:
        if st.button(f"Trip {trip['trip_id']}", key=f"trip_{trip['trip_id']}",
                     help=f"Partition: {trip['partition']}", use_container_width=True):
            st.session_state["selected_trip"] = trip["trip_id"]
            st.session_state.pop("result", None)
            st.session_state.pop("network", None)

selected = st.session_state.get("selected_trip")
if selected is None:
    st.info("Choose a trip above to run the prediction.")
    st.stop()

st.markdown(f"**Selected trip:** `{selected}`")

if "result" not in st.session_state or st.session_state.get("result_trip") != selected:
    with st.spinner(f"Running prediction for trip {selected} on the backend..."):
        try:
            st.session_state["result"] = api_post(api_url, "/api/predict", {"trip_id": selected})
            st.session_state["result_trip"] = selected
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("result", None)
            st.error(explain_error(exc, api_url))
            st.stop()
    try:
        st.session_state["network"] = api_get(api_url, f"/api/trips/{selected}/network")
    except Exception as exc:  # noqa: BLE001
        st.session_state["network"] = None
        st.warning(f"Road-network layer unavailable: {explain_error(exc, api_url)}")

result = st.session_state["result"]
network = st.session_state.get("network")
metrics = result["metrics"]

# ----------------------------------------------------------------- results
st.subheader("Evaluation results")
clf, hmm = metrics["classifier_only"], metrics["hmm_viterbi"]
rows = [
    ("Observation Top-1 accuracy", "top1_accuracy", "pct"),
    ("Macro F1 (observation level)", "macro_f1", "num"),
    ("Route arc F1", "route_edge_f1", "pct"),
    ("Route LCS recall (ordered)", "route_lcs_recall", "pct"),
    ("Exact sequence match", "exact_sequence_match", "pct"),
    ("Route continuity", "continuity", "pct"),
    ("Predicted route length (arcs)", "predicted_route_length", "int"),
]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.4f}"


table = pd.DataFrame(
    [{"Metric": label, "Classifier only": fmt(clf.get(key), kind), "XGBoost + HMM/Viterbi": fmt(hmm.get(key), kind)}
     for label, key, kind in rows]
)
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.table(table.set_index("Metric"))
with c2:
    st.metric("Observations", metrics["observations"])
    st.metric("Candidate coverage", fmt(metrics["candidate_coverage"], "pct"),
              help="Share of observations whose correct road arc is among the candidates.")
with c3:
    st.metric("Top-1 gain from HMM", f"{(hmm['top1_accuracy'] - clf['top1_accuracy']) * 100:+.2f} pts")
    st.metric("Backend time", f"{result['timings_ms']['total_model_time']:.0f} ms")
    st.caption(f"Request ID: {result.get('request_id')}")

with st.expander("Timing and model details"):
    st.json({"timings_ms": result["timings_ms"], "model": result["model"], "trip": result["trip"]})

# ----------------------------------------------------------------- map
st.subheader("Map")
show_hmm = st.radio("Predicted route to highlight", ["XGBoost + HMM/Viterbi", "Classifier only", "Both"],
                    horizontal=True, index=0)

obs = result["observations"]
lats = [o["lat"] for o in obs]
lons = [o["lon"] for o in obs]
center = [sum(lats) / len(lats), sum(lons) / len(lons)]
fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)

gps_layer = folium.FeatureGroup(name="Original GPS observations", show=True)
for o in obs:
    folium.CircleMarker(location=[o["lat"], o["lon"]], radius=3, color="#1f2937",
                        fill=True, fill_opacity=0.9, weight=1,
                        tooltip=f"obs {o['obs_idx']} (t={o['timestamp']:.0f}s)").add_to(gps_layer)
gps_layer.add_to(fmap)
folium.PolyLine([[o["lat"], o["lon"]] for o in obs], color="#1f2937", weight=1, opacity=0.4,
                tooltip="GPS track").add_to(fmap)

if network and network.get("candidate_arcs"):
    net_layer = folium.FeatureGroup(name="Candidate road network", show=True)
    for arc in network["candidate_arcs"]:
        folium.PolyLine(arc["segment"], color="#9ca3af", weight=3, opacity=0.6,
                        tooltip=f"candidate arc {arc['edge_id']}").add_to(net_layer)
    net_layer.add_to(fmap)

if show_hmm in ("Classifier only", "Both"):
    layer = folium.FeatureGroup(name="Predicted route: classifier only", show=True)
    for seg in result["classifier_only"]["route_segments"]:
        folium.PolyLine(seg, color="#f97316", weight=5, opacity=0.85).add_to(layer)
    layer.add_to(fmap)

if show_hmm in ("XGBoost + HMM/Viterbi", "Both"):
    layer = folium.FeatureGroup(name="Predicted route: XGBoost + HMM/Viterbi", show=True)
    for seg in result["hmm_viterbi"]["route_segments"]:
        folium.PolyLine(seg, color="#2563eb", weight=5, opacity=0.85).add_to(layer)
    layer.add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

legend = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
     padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px; color: #111;">
  <b>Legend</b><br>
  <span style="color:#1f2937">&#9679;</span> GPS observations<br>
  <span style="color:#9ca3af">&#9632;</span> Candidate road network<br>
  <span style="color:#f97316">&#9473;</span> Classifier-only route<br>
  <span style="color:#2563eb">&#9473;</span> XGBoost + HMM/Viterbi route
</div>"""
fmap.get_root().html.add_child(folium.Element(legend))
st_folium(fmap, height=560, use_container_width=True, returned_objects=[])

st.caption("Ground-truth reference routes are evaluated on the server and are not displayed.")
