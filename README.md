# GridSense (Streamlit) - all files in ONE folder
app.py, requirements.txt, README.md,
SecureElite_DeviceID19_KW_gt_1.xlsx   (running data only -> prediction + load balance)
SecureElite_DeviceID19_sorted.parquet (full minute data incl. zeros -> overview + live replay; same data as your sorted .xlsx, converted so it loads in 1 s instead of ~85 s)

Run:    pip install -r requirements.txt   then   streamlit run app.py
Deploy: put these 5 files in the root of a GitHub repo -> share.streamlit.io -> main file app.py
(If SecureElite_DeviceID19_sorted.parquet is missing, the app falls back to SecureElite_DeviceID19_sorted.xlsx placed in the same folder - slow to read.)
