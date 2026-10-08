import subprocess
import shutil
import os
from pathlib import Path

print("Running A (Base)...")
subprocess.run(["python", "verify_cartpole.py"], check=True)

print("Preparing B (temp_min=0.1)...")
with open("verify_cartpole.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("temp_min=0.3", "temp_min=0.1")
code = code.replace('run_A_base.json', 'run_B_temp01.json')
code = code.replace('run_A_base_X.npy', 'run_B_temp01_X.npy')
with open("verify_cartpole.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Running B...")
subprocess.run(["python", "verify_cartpole.py"], check=True)

print("Preparing C (No Accumulation)...")
code = code.replace("temp_min=0.1", "temp_min=0.3")
code = code.replace('run_B_temp01.json', 'run_C_noaccum.json')
code = code.replace('run_B_temp01_X.npy', 'run_C_noaccum_X.npy')
with open("verify_cartpole.py", "w", encoding="utf-8") as f:
    f.write(code)

with open("neurobot/agent.py", "r", encoding="utf-8") as f:
    agent_code = f.read()

agent_code_C = agent_code.replace(
    "self._E_accum += np.outer(pre.astype(float), delta)",
    "self._E_accum = np.outer(pre.astype(float), delta)"
)
with open("neurobot/agent.py", "w", encoding="utf-8") as f:
    f.write(agent_code_C)

print("Running C...")
subprocess.run(["python", "verify_cartpole.py"], check=True)

print("Preparing D (Hebbian without homeostasis)...")
code = code.replace('run_C_noaccum.json', 'run_D_hebbian.json')
code = code.replace('run_C_noaccum_X.npy', 'run_D_hebbian_X.npy')
with open("verify_cartpole.py", "w", encoding="utf-8") as f:
    f.write(code)

agent_code_D = agent_code.replace(
    "delta = indicator - probs                    # (M,)\n        self._E_accum += np.outer(pre.astype(float), delta)",
    "self._E_accum += np.outer(pre.astype(float), indicator)"
)
agent_code_D = agent_code_D.replace(
    "dX = -self.eta * self._E_accum * signal",
    "dX = -self.eta * self._E_accum"
)
with open("neurobot/agent.py", "w", encoding="utf-8") as f:
    f.write(agent_code_D)

print("Running D...")
subprocess.run(["python", "verify_cartpole.py"], check=True)

print("Restoring original agent.py and verify_cartpole.py...")
with open("neurobot/agent.py", "w", encoding="utf-8") as f:
    f.write(agent_code)

code = code.replace('run_D_hebbian.json', 'run_A_base.json')
code = code.replace('run_D_hebbian_X.npy', 'run_A_base_X.npy')
with open("verify_cartpole.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Done running all configs!")
