"""Four-screenshot video analysis.

Workflow (see docs/video_analysis/PLAN.md):
    upload + probe (media.py) -> plan 4 random frames (plan.py) -> decode them in order (extract.py)
    -> propose an outdoor ROI (roi_proposal.py) -> operator confirmation -> existing YOLO on the 4 frames
    -> annotated images + CSV (report.py) -> Results Dashboard (reports.py)

Only litter detection. Local storage only (no cloud input or output).
"""

SAMPLE_COUNT = 4   # mandatory: every job analyses exactly four screenshots
