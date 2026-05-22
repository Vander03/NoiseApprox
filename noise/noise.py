import os, json
from collections import defaultdict
import random
from tqdm import tqdm
from qiskit_ibm_runtime.models import BackendProperties
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
import pennylane as qml
from pennylane import AmplitudeEmbedding

DEBUGGING = False

"""
NOTE:
Fake backends are not intended to represent real backends error ranges.
This means that theyre good for testing while I dont have the historical backends working
So I can use these along with live calibration profiles for now to test the noise training, 
and then extend to historical noise profiles
"""

class noise:
    def __init__(self, fake=None, hist_count=None):
        self.fake = fake # determines if the model uses fake noise profiles to train
        self.hist_count = hist_count # the number of historical noise profiles to load for each backend

        # self.holdout_profiles = [
        #     "hist_ibm_kingston_2026-04-15.json",
        #     "hist_ibm_kingston_2026-04-11.json",
        #     "hist_ibm_kingston_2026-04-12.json",
        #     "hist_ibm_fez_2026-04-16.json",
        #     "hist_ibm_fez_2026-04-04.json",
        #     "hist_ibm_fez_2026-04-10.json",
        #     "hist_ibm_marrakesh_2026-04-12.json",
        #     "hist_ibm_marrakesh_2026-04-08.json",
        #     "hist_ibm_marrakesh_2026-04-10.json",
        #     "hist_ibm_marrakesh_2026-04-14.json",
        # ]
        self.holdout_profiles = [
            "hist_ibm_kingston_2026-04-24.json",
            "hist_ibm_kingston_2026-04-13.json",
            "hist_ibm_kingston_2026-03-23.json",
            "hist_ibm_kingston_2026-01-29.json",
            "hist_ibm_kingston_2026-03-18.json"
        ]

    def load_calibration_data(self, load_prof=None, limit_backends=None):
        files = [f for f in os.listdir("calibrations") if f.endswith(".json")]
        
        # pre-filter filenames before loading from disk
        filtered_files = []
        for filename in files:
            is_fake = filename.startswith("fake_")
            is_hist = filename.startswith("hist_")
            if is_fake and not self.fake:
                continue
            if is_hist and not self.hist_count:
                continue
            if limit_backends and limit_backends not in filename:
                continue
            filtered_files.append(filename)

        # if hist, pre-select which files to load before loading them
        if self.hist_count:
            hist_files = [f for f in filtered_files if f.startswith("hist_")]
            non_hist_files = [f for f in filtered_files if not f.startswith("hist_")]
            
            # group by backend name
            by_backend = defaultdict(list)
            for f in hist_files:
                backend = "_".join(f.replace("hist_", "").split("_")[:-1])  # extract backend name
                by_backend[backend].append(f)

            selected_hist = []
            for backend, files in by_backend.items():
                files_sorted = sorted(files, reverse=True)  # most recent first
                selected_hist.extend(files_sorted[:self.hist_count])
            
            filtered_files = non_hist_files + selected_hist

        # if load_prof is not None, load the profiles provided
        if load_prof:
            if isinstance(load_prof, str):
                load_prof = [load_prof]  # wrap single filename in a list
            filtered_files = list(load_prof)

        # now load only the selected files
        profiles = []
        for filename in tqdm(filtered_files, desc="Loading calibration profiles"):
            filepath = os.path.join("calibrations", filename)
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                data = json.load(f)
            prof = noise.build_backend(data, filename)
            if prof is not None:
                profiles.append(prof)

        return profiles

    @staticmethod
    def build_backend(data, filename):
        props = BackendProperties.from_dict(data["properties"])
        try:
            noise_model = NoiseModel.from_backend_properties(props, thermal_relaxation=False)
            if DEBUGGING: print(f"{data['backend']} - (thermal relaxation skipped - no frequency data)")
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
            return
        csc = noise.compute_csc_from_props(props)
        return ({
                "filename": filename,
                "backend": data["backend"],
                "date": data["date"],
                "noise_model": noise_model,
                "props": props,
                "csc": csc
            })

    @staticmethod
    def compute_csc_from_props(props):
        qubits = []
        for qubit_idx in range(len(props.qubits)):
            readout_error = props.readout_error(qubit_idx) or 0.0
            gate_errors = []
            for gate in props.gates:
                if qubit_idx in gate.qubits:
                    try:
                        err = props.gate_error(gate.gate, gate.qubits)
                        if err is not None:
                            gate_errors.append(err)
                    except Exception:
                        pass  # skip gates with no error property (e.g. reset)
            qubits.append((gate_errors, readout_error))
        return noise.CSC(qubits)

    @staticmethod
    def GSC(gate_errors, readout_error):
        """
        gate_errors: list of per-gate error probabilities for this qubit
        readout_error: single readout error probability for this qubit
        """
        gate_correctness = 1.0
        for gate_err in gate_errors:
            gate_correctness *= (1 - gate_err)
        return gate_correctness * (1 - readout_error)

    @staticmethod
    def CSC(qubits):
        """
        qubits: list of (gate_errors, readout_error) per qubit
        """
        return sum(noise.GSC(gate_errors, readout_error) for gate_errors, readout_error in qubits) / len(qubits)

    # def MSE(ideal_emb, perturbed_emb):
    #     """
    #     Computes MSE between two embedding arrays for all embeddings in a batch
    #     """
    #     return sum([(prediction - target) ** 2 for prediction, target in zip(ideal_emb, perturbed_emb)])

