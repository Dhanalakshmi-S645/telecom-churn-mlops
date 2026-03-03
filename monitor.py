import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import warnings
warnings.filterwarnings("ignore")

from src.preprocess import get_data_splits


def generate_drift_report():
    print("Generating Evidently AI monitoring reports...")

    X_train, X_test, y_train, y_test = get_data_splits()
    model = joblib.load("models/best_model.pkl")

    ref_df = X_train.copy()
    ref_df["target"] = y_train.values
    ref_df["prediction"] = model.predict(X_train)

    cur_df = X_test.copy()
    cur_df["target"] = y_test.values
    cur_df["prediction"] = model.predict(X_test)

    os.makedirs("reports", exist_ok=True)

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset, DataQualityPreset
        from evidently import ColumnMapping

        column_mapping = ColumnMapping(target="target", prediction="prediction")

        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=ref_df.drop(columns=["target", "prediction"]),
            current_data=cur_df.drop(columns=["target", "prediction"])
        )
        drift_report.save_html("reports/data_drift_report.html")
        print("Data drift report saved")

        perf_report = Report(metrics=[ClassificationPreset()])
        perf_report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
        perf_report.save_html("reports/model_performance_report.html")
        print("Model performance report saved")

    except Exception as e:
        print(f"Evidently import issue, using fallback: {e}")
        generate_manual_reports(ref_df, cur_df)

    print("\nAll monitoring reports generated in reports/ folder")


def generate_manual_reports(ref_df, cur_df):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    feature_cols = [c for c in ref_df.columns if c not in ["target", "prediction"]]

    drift_rows = ""
    drifted = 0
    for col in feature_cols:
        ref_mean = ref_df[col].mean()
        cur_mean = cur_df[col].mean()
        ref_std  = ref_df[col].std() + 1e-9
        z_score  = abs((cur_mean - ref_mean) / ref_std)
        drifted_flag = "YES" if z_score > 0.3 else "No"
        color = "#FADBD8" if z_score > 0.3 else "#D5F5E3"
        if z_score > 0.3:
            drifted += 1
        drift_rows += f'<tr style="background:{color}"><td>{col}</td><td>{ref_mean:.4f}</td><td>{cur_mean:.4f}</td><td>{z_score:.4f}</td><td>{drifted_flag}</td></tr>'

    acc  = accuracy_score(cur_df["target"], cur_df["prediction"])
    f1   = f1_score(cur_df["target"], cur_df["prediction"])
    prec = precision_score(cur_df["target"], cur_df["prediction"])
    rec  = recall_score(cur_df["target"], cur_df["prediction"])

    html = f"""<!DOCTYPE html><html><head><title>MLOps Monitoring Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;background:#f8f9fa}}
h1{{color:#1F4E79}}h2{{color:#2E86C1;border-bottom:2px solid #2E86C1;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;background:white;margin-bottom:30px}}
th{{background:#1F4E79;color:white;padding:12px;text-align:left}}
td{{padding:9px 12px;border-bottom:1px solid #ddd}}
.box{{display:inline-block;background:white;border-radius:8px;padding:20px 30px;margin:10px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.1)}}
.val{{font-size:2.2em;font-weight:bold;color:#1F4E79}}
.lbl{{color:#666;font-size:.9em}}
.banner{{background:#1F4E79;color:white;padding:20px 30px;border-radius:10px;margin-bottom:30px}}
</style></head><body>
<div class="banner"><h1 style="color:white;margin:0">Telecom Churn - MLOps Monitoring Dashboard</h1>
<p style="margin:5px 0 0;color:#AED6F1">Data Drift Detection + Model Performance Report</p></div>
<h2>Data Drift Summary</h2>
<p><b>{drifted}/{len(feature_cols)}</b> features show drift (Z-score &gt; 0.3)</p>
<table><tr><th>Feature</th><th>Reference Mean</th><th>Current Mean</th><th>Z-Score</th><th>Drifted?</th></tr>
{drift_rows}</table>
<h2>Model Performance on Current Data</h2>
<div>
<div class="box"><div class="val">{acc:.3f}</div><div class="lbl">Accuracy</div></div>
<div class="box"><div class="val">{f1:.3f}</div><div class="lbl">F1 Score</div></div>
<div class="box"><div class="val">{prec:.3f}</div><div class="lbl">Precision</div></div>
<div class="box"><div class="val">{rec:.3f}</div><div class="lbl">Recall</div></div>
</div>
<br><p style="color:#999;font-size:.85em">Reference: {len(ref_df)} samples | Current: {len(cur_df)} samples</p>
</body></html>"""

    with open("reports/data_drift_report.html", "w") as f:
        f.write(html)
    print("Data drift + performance report saved to reports/data_drift_report.html")


if __name__ == "__main__":
    generate_drift_report()