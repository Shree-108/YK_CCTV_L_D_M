"""Four-screenshot video analysis.

Workflow (see docs/video_analysis/PLAN.md):
    upload + probe (media.py) -> plan 4 frames (plan.py) -> decode them in order (extract.py)
    -> YOLO on the 4 full frames -> annotated images + CSV (report.py, detection_report.py)
    -> standard litter outputs (backend/services/analysis_outputs.py) -> Results Dashboard (reports.py)

Only litter detection. Local storage only (no cloud input or output).
"""

SAMPLE_COUNT = 4   # mandatory: every job analyses exactly four screenshots
