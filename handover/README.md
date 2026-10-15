# LDM handover package (2026-10-12)

**Status: READY FOR CONTROLLED DEMONSTRATION** (local PC, current production model `v3_store_cctv`). Not ready for a
limited pilot; see the evidence in `UAT_REPORT.md`.

| # | Deliverable | File |
|---|---|---|
| 1 | Final UAT report (decision, limitations, next steps) | [UAT_REPORT.md](UAT_REPORT.md) |
| 2 | Test-case checklist and results (61 cases) | [TEST_CASES.md](TEST_CASES.md) |
| 3 | Defect and blocker register | [DEFECTS_AND_BLOCKERS.md](DEFECTS_AND_BLOCKERS.md) |
| 4 | Application setup guide | [../deploy/SETUP.md](../deploy/SETUP.md) |
| 5 | Startup / shutdown guide (manager and QA) | [../deploy/README.md](../deploy/README.md) · troubleshooting: [../deploy/TROUBLESHOOTING.md](../deploy/TROUBLESHOOTING.md) |
| 6 | Deployment, backup and rollback guide | [DEPLOYMENT_AND_ROLLBACK.md](DEPLOYMENT_AND_ROLLBACK.md) |
| 7 | Model evaluation summary | [MODEL_EVALUATION_SUMMARY.md](MODEL_EVALUATION_SUMMARY.md) |
| 8 | CCTV integration status and vendor dependencies | [CCTV_INTEGRATION_STATUS.md](CCTV_INTEGRATION_STATUS.md) → [../docs/CCTV_RECORDING_RETRIEVAL.md](../docs/CCTV_RECORDING_RETRIEVAL.md) |
| 9 | Known limitations | `UAT_REPORT.md` → Known limitations |
| 10 | Recommended next steps for a controlled pilot | `UAT_REPORT.md` → Next steps |

Supporting documents:

- Monitoring and continuous improvement: [../docs/LDM_MONITORING.md](../docs/LDM_MONITORING.md); weekly report:
  `py -m monitoring.ldm_monitor`; approved-change log: [../governance/model_change_log.csv](../governance/model_change_log.csv).

- Dataset work: [../docs/LDM_DATASET_GUIDE.md](../docs/LDM_DATASET_GUIDE.md) and
  [../docs/LDM_ANNOTATION_GUIDE.md](../docs/LDM_ANNOTATION_GUIDE.md).
- Fine-tuning plan (not run): [../docs/LDM_FINETUNE_PLAN.md](../docs/LDM_FINETUNE_PLAN.md).
- Evidence: `evidence/`. Test outputs and scripts are included; CCTV images are git-ignored for privacy.

Quick start:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1
```

Then open http://localhost:3010. To stop, run `deploy\stop-ldm.cmd`.
