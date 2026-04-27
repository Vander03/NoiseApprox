import json
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.models import BackendProperties
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_aer.noise import NoiseModel

def load_noise_model_from_snapshot(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    service = QiskitRuntimeService()
    # backend = service.backend(data["backend"])
    
    props = BackendProperties.from_dict(data["properties"])
    noise_model = NoiseModel.from_backend_properties(props)
    
    return noise_model, props

# Test it
noise_model, props = load_noise_model_from_snapshot("calibrations/fake_toronto_2026-04-09.json")
print(noise_model)