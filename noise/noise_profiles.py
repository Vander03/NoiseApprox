import os, json
import pandas as pd
from pandas import DataFrame
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
from datetime import datetime, timedelta

# stackoverflow
# https://quantumcomputing.stackexchange.com/questions/40011/how-to-download-historical-calibration-data-from-ibm-quantum-devices

service = QiskitRuntimeService()

def fetch_fake_calibration_history():
    print("Fake Backends")
    fake_backends = FakeProviderForBackendV2()
    # print([(b.name, b.num_qubits) for b in fake_backends.backends()])
    snapshots = []
    for backend in fake_backends.backends():
        props = backend.properties()
        snapshots.append({
            "date": props.last_update_date.isoformat(),
            "backend": backend.name,
            "properties": props.to_dict()
        })
        print(f"✓ Got properties for {backend.name} - {props.last_update_date.isoformat()}")
    return snapshots


# fetches a list of snapshots for all the quantum devices I have access too, across a range of x days apart
def fetch_calibration_history(num_snapshots=5, days_apart=3):
    available_backends = service.backends()
    snapshots = []
    base_date = datetime.now()
    
    calibrations_dir = os.path.join(os.path.dirname(__file__), '..', 'calibrations')
    existing = set(os.listdir(calibrations_dir)) if os.path.exists(calibrations_dir) else set()


    for backend in available_backends:
        for i in range(num_snapshots):
            target_date = base_date - timedelta(days=i * days_apart)
            try:
                props = backend.properties(datetime=target_date)
                if props is None:
                    continue
                    
                date_str = props.last_update_date.strftime("%Y-%m-%d")
                filename = f"hist_{backend.name}_{date_str}.json"
                
                if filename in existing:
                    print(f"↷ Skipping {filename} (already exists)")
                    continue
                
                snapshots.append({
                    "date": props.last_update_date.isoformat(),
                    "backend": backend.name,
                    "properties": props.to_dict()
                })
                existing.add(filename)
                print(f"✓ Got properties for {backend.name} - {date_str}")
                    
            except Exception as e:
                print(f"✗ {backend.name} - {target_date.strftime('%Y-%m-%d')}: {e}")
    
    return snapshots

def fetch_current_data():
    available_backends = service.backends()
    
    snapshots = []
    base_date = datetime.now()
    
    for backend in available_backends:
        try:
            props = backend.properties()
            if props is not None:
                snapshots.append({
                    "date": props.last_update_date.isoformat(),
                    "backend": backend.name,
                    "properties": props.to_dict()
                })
                print(f"✓ Got properties for {backend.name} - {props.last_update_date.isoformat()}")
        except Exception as e:
            print(f"✗ {backend.name} - {props.last_update_date.isoformat()}: {e}")
    
    return snapshots

def save_calibration_library(num_snapshots=2, days_apart=1, fake=False, hist=False):
    os.makedirs("calibrations", exist_ok=True)
    
    if fake:
        snapshots = fetch_fake_calibration_history()
    elif hist:
        snapshots = fetch_calibration_history(num_snapshots, days_apart)
    else:
        snapshots = fetch_current_data()
    
    today = datetime.now().strftime("%Y-%m-%d")
    for snap in snapshots:
        date_str = snap["date"][:10]
        tag = "hist_"
        if hist:
            path = f"../calibrations/hist_{snap['backend']}_{date_str}.json"
        else:
            path = f"../calibrations/{snap['backend']}_{date_str}.json"
        with open(path, "w") as f:
            json.dump(snap, f, indent=2, default=str)
        print(f"Saved: {path}")



# save_calibration_library(num_snapshots=5, fake=False, hist=False)
# save_calibration_library(num_snapshots=5, fake=True, hist=False)
save_calibration_library(num_snapshots=600, fake=False, hist=True)

