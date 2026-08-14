import json
import os
from qiskit_ibm_runtime import QiskitRuntimeService
    
print(os.environ["QISKIT_IBM_TOKEN"])
QiskitRuntimeService.save_account(token=os.environ["QISKIT_IBM_TOKEN"], overwrite=True, channel="ibm_quantum_platform")

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backends = service.backends()   # requires a valid token
    print(f"Authenticated OK — {len(backends)} backends available")
    print([b.name for b in service.backends() if b.status().operational])
except Exception as e:
    print(f"Authentication failed: {type(e).__name__}: {e}")
