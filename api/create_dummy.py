"""Create models/dummy.joblib for API and dashboard demos."""
from pathlib import Path
import joblib
from api.demo_model import DemoModel

path = Path(__file__).resolve().parents[1] / "models" / "dummy.joblib"
path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(DemoModel(), path)
print(f"Saved demo model: {path}")

