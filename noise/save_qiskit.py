from qiskit_ibm_runtime import QiskitRuntimeService
 
QiskitRuntimeService.save_account(
  token="", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
  set_as_default=True, # Optional
  overwrite=True, # Optional
)