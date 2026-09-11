# Petrochronology-Imaging-Streamlit-Application


## Import

Every fresh import requires **sample ID, mineral ID, run ID, X pixel size (µm), and Y pixel size (µm)**. Fields start empty. There are no sample/mineral/run filename inference rules.

- **Header-bearing tables:** choose raster channels, a point layer, or an analysis table. Select multiple value channels when rasterizing. X/Y columns are physical coordinates in µm. They must align with the explicit raster sizes; coordinate gaps become missing cells. Import irregular coordinates as a point layer.
- **Headerless matrix CSV:** select aligned channels for one explicit sample/mineral/run, edit the channel names, and enter both pixel dimensions and the physical origin. Matrices must have matching shapes. Cropping preserves the original spatial offsets. Optional paired X/Y reference matrices preserve physical coordinates; complete grids and sparse affine references are supported.
- **Probe DAT / Surfer 7 DSRB:** file metadata is displayed for review. Explicit origins and dimensions set the imported regular grid. Rotated Surfer grids are rejected.
- Import each sample/mineral/run separately. An existing layer/table key is rejected rather than silently overwritten.

The input coordinates and pixel footprint are distinct concepts for point layers. Pixel sizes are retained in project metadata; grain areas use both dimensions. A single-row or single-column map keeps its declared footprint.

## Main additions

- BSE, CHUR, CI, and MORB REE normalization with the exact source constants, arithmetic overall/group means, individual-row plots, distance zones, six envelopes, IQR filtering, and data exports. CHUR and CI intentionally have identical values because the reference does.
- Actual Gaussian KDE curves and 2D densities, correlation ranking, PCA biplots and driver rankings, custom groups, PCA exports, and validated ternary fractions with group/symbol styling.
- Feret boundary ellipses, full-ellipse radial extraction, moved-center axis refits, and bilinear/nearest line sampling.
- Workspace tools for calculated maps, all three reference pixel exclusion scopes/undo, saved formula and alias replay, sample/mineral/run rename, channel rename, and calibration changes.
- A population concordia/discordia page using the reference numerical routines, including fixed common-Pb intercepts, shared external uncertainty, MSWD expansion, and configurable decay constants.
- Project format v2 uses typed JSON tables so empty tables and identifiers such as `001` survive save/reload. Streamlit v1 project files can still be read; information already lost by old CSV saves cannot be recovered automatically.

Use **Population concordia and discordia fits** for the reference Wetherill axis convention: X = 207Pb/235U, Y = 206Pb/238U. The retained Advanced U–Pb page labels its opposite axis order explicitly.

Workspace exclusions, alias merges, and calibration edits invalidate affected derived data. Re-run detection/extraction afterward. Undo restores the prior analysis snapshot. Saved formula/alias definitions replay on future imports and on request, with absent operands reported as pending. Recompute downstream grain/selection products after changing their inputs.

Data are held by the Streamlit server, not solely in the browser. Save a project before ending the session. Offline HTML figures embed Plotly. Each plot also offers PNG/SVG/PDF generation through Kaleido, which requires Chrome/Chromium installed on the server (for example, `plotly_get_chrome`). Static generation failed during this validation because the browser closed before connection; these formats are not runtime-verified in this delivery.






