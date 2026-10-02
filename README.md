# Petrochronology Imaging

Mineral maps, pixel domains, grains and spatial analysis.

## Local

Install Python 3.13. Extract the folder, then open **Start Mac.command** or **Start Windows.bat**. The first run installs dependencies. Keep the terminal open while using the app.

If macOS blocks the launcher, run `python3 launch.py` in Terminal from this folder.

## Streamlit Cloud

Upload this folder’s contents to your GitHub repository. Create a Streamlit app using **app.py** as the entry point and Python **3.13** in Advanced settings. Keep `requirements.txt` and `packages.txt` beside `app.py`.

[Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Use

Import → Maps → Domains → Plots or Grains → Spatial → Save.

Save drawings before preparing a project snapshot, then click Download. Keep a copy on your computer. Large datasets need enough RAM on whichever computer hosts the app.

Use a fresh folder for this release. `core/` contains shared calculations and controls; `pages/` contains the screens. Both are required.
