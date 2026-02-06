from pathlib import Path
import pandas as pd
from epinet.paths import project_root

def main():
    out_dir = project_root() / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"cascade_id": 0, "node_id": 1, "infection_time": 0.0},
            {"cascade_id": 0, "node_id": 2, "infection_time": 1.0},
            {"cascade_id": 0, "node_id": 3, "infection_time": 2.0},
            {"cascade_id": 1, "node_id": 2, "infection_time": 0.0},
            {"cascade_id": 1, "node_id": 1, "infection_time": 1.5},
            {"cascade_id": 1, "node_id": 3, "infection_time": 3.0},
        ]
    )

    path = out_dir / "toy_cascades.csv"
    df.to_csv(path, index=False)
    print("Wrote", path)
    print(df.head())

if __name__ == "__main__":
    main()
