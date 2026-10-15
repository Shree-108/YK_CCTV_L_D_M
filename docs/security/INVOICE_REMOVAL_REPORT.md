# Invoice extraction removed from the LDM application

Branch `security/step1-local-only`, based on 385a65d. This cleanup is committed on its own: the Step 3 login-screen work that shares the working tree is not part of it (in `backend/main.py` only the invoice import and router registration are included).

## Recovery checkpoint

- Branch `checkpoint/pre-invoice-removal-20261015` points at 385a65d, the last commit with the full invoice feature.
- The uncommitted Step 3 work as it was before this cleanup is saved in `.git/ldm-checkpoints/pre-invoice-removal-20261015/`:
  - `step3_uncommitted.patch`;
  - `step3_untracked.tar`.
- To recover any invoice file: `git checkout checkpoint/pre-invoice-removal-20261015 -- <path>`.

## Removed

| Area | What |
|---|---|
| API routes | `GET /api/invoices/engine`, `POST /api/invoices/extract` (OpenAPI: 71 → 69 operations; the `invoices` tag is gone) |
| Backend | `backend/api/routes_invoice.py`; import and `include_router` removed from `backend/main.py` |
| Package | `invoice_extractor/` (17 files, including its `requirements.txt`) |
| Frontend | the `/invoices` page (`app/invoices/page.tsx`), `frontend/features/invoices/InvoiceExtractionPage.tsx`, the "Invoice Extraction" sidebar/command-palette item (`nav.ts`), `invoiceEngine`/`extractInvoices` calls (`endpoints.ts`), invoice types (`types.ts`) |
| Scripts | `run_invoice_extraction.bat`, `setup_invoice_extractor.bat` |
| Tests | `tests/test_invoice_extractor.py`; the `/api/invoices/engine` case in `tests/test_local_only.py` |
| Dependencies (`requirements.txt`) | `onnxruntime`, `pyclipper`, `shapely`, `omegaconf`, `colorlog`, `tqdm`, `pytesseract` and the rapidocr note. Before removal it was confirmed that no LDM code imports any of them and that, among installed packages, only `rapidocr` requires them (ultralytics does not). |

**Added:** `tests/test_app_routes.py`. It fails if any invoice route reappears in the OpenAPI schema or responds.

## Kept on purpose

- **`invoices/` folder:**
  - `input/` and `output/` contain only `.gitkeep`, so there is no real invoice data;
  - `samples/` holds 4 synthetic sample images.
  - Not deleted, as instructed. It can be removed later if you decide to.
- **`.gitignore` invoice rules:** these keep anything placed in `invoices/` out of git while that folder exists.
- **Installed Python packages** (`rapidocr`, `onnxruntime`, …): still installed in the Python environment, because only the requirement lines were removed. Uninstalling them is optional and was not done.
- **Historical records:** Step 0–3 security reports and git history (commits up to 385a65d and the checkpoint branch) still mention invoice extraction; they are kept as the record of what was tested at the time.
- **Database:** the live database has no invoice tables. Its tables are cameras, detections, confirmed_objects, alerts, settings, system_events, models, hard_examples, reviews, captures and jobs. Nothing was dropped.

## Verification

Checks were run twice: on this commit's content alone (385a65d plus the cleanup, without Step 3) and on the working tree (cleanup plus uncommitted Step 3).

- **Backend tests (isolated storage):** exit 0 both times.
  - This commit alone: `131 passed, 38 skipped`.
  - Working tree: `133 passed, 38 skipped`; the 2 extra tests are Step 3's.
  - With Step 3, the count before the cleanup was 166. The change is −34 invoice tests, −1 local-only invoice case and +2 new route tests.
  - The 38 skipped tests are an **unresolved coverage limitation**: integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2. 37 are skipped with the reason "trained model/test images not present" and 1 with "no test images".
- **TypeScript:** `tsc --noEmit` exit 0 both times.
  - Stale generated route types in the gitignored `.next/dev/types` still pointed to the deleted page. They were removed, and `next dev` regenerates them.
- **Production build:** compiled successfully both times with 23 pages (previously 24). There is no `/invoices` route and no invoice page output.
- **Running app:** the FastAPI app registers no invoice route, and `/api/invoices/*` returns 404. No `invoice*` module is loaded.
- **Browser, isolated stack** (temporary storage, login on, synthetic operator account; all removed afterwards):
  - sign-in works;
  - the dashboard shows no invoice link;
  - `POST /api/detect/image` returned 200 with model `v3_store_cctv`;
  - `POST /api/detect/video` returned 202, and the job reached `done`;
  - the Image Detection, Video Detection, Alerts, Cameras and Live CCTV pages load;
  - `/api/alerts`, `/api/cameras`, `/api/analytics`, `/api/jobs` and `/api/health` returned 200;
  - `/invoices` shows 404;
  - Sign out works;
  - the test password and session cookie appear in no log.
- **Integrity:**
  - live DB fingerprint `f7c83c2dafd7ea24` unchanged;
  - model sha256 `d1de9cb6…` unchanged;
  - the live DB still has 0 settings rows, the same as the baseline;
  - no settings, detection, CCTV, local-only or auth file was changed by this cleanup;
  - no real `storage/auth.db`;
  - `main` (95c151b) and the snapshot (d0ea84d) are unchanged;
  - nothing pushed.
