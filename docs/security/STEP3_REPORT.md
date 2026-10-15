# API security — Step 3 report: login screen and session cookie (frontend)

Status: **implemented and verified; not committed** (awaiting approval). Branch `security/step1-local-only`, based on 385a65d.
Authentication stays **off by default** (`LM_AUTH_MODE=off`). **API routes are still not protected.** That is Step 4.

> **Later change:** invoice extraction has since been removed from the LDM application (see `INVOICE_REMOVAL_REPORT.md`). The invoice checks below record what was tested at the time of Step 3.

## What changed

| File | Change |
|---|---|
| `frontend/api/auth.ts` (new) | `fetchMe`, `login` and `logout` for `/api/auth/*`. Every request uses `credentials: 'include'` and `cache: 'no-store'`. Each status code maps to a fixed message (401 invalid, 429 locked with minutes taken from `Retry-After`, 403 local-only, 404 login disabled, network failure unreachable). Raw server errors are never shown. |
| `frontend/features/auth/AuthGate.tsx` (new) | Wraps the app. **Auth off:** renders the app exactly as before. **Auth on:** shows the login screen until `/me` reports a session. Re-checks `/me` every 60 s and on focus or visibility change. Shows "Your session has expired" when a session ends on the server. If the server is unreachable, the app renders as before and retries every 15 s. The gate is UI-only. |
| `frontend/features/auth/LoginScreen.tsx` (new) | Sign-in card built from the existing brand wordmark and UI primitives. The password is read from the form only at submit and the field is cleared after every attempt. It is never held in React state or storage, and never logged. Shows a loading state ("Signing in…"), errors with `role=alert`, and notices for "expired" and "signed out". |
| `app/layout.tsx` | `<AuthGate>` wraps the existing `AppProvider` and `AppShell`. |
| `frontend/components/shell/AppShell.tsx` | When signed in, the top bar shows the username, role and a **Sign out** button. The admin-token chip is unchanged. |
| `frontend/api/client.ts` | `credentials: 'include'` on API fetches and authenticated downloads. |
| `frontend/stores/app.tsx` | The live-event stream uses `EventSource(..., { withCredentials: true })`. |
| `backend/main.py` | CORS: `allow_credentials=True`, which is required for the browser to send the cookie, still limited to the explicit `CORS_ORIGINS` list (never `*`). Allowed headers are named explicitly. `Retry-After` is exposed. |
| `backend/auth/__main__.py` | Bug fix: `--password-stdin` now strips the UTF-8 byte-order mark that Windows PowerShell 5.1 adds to piped text. Without it, the stored password silently differed from the one typed. |
| `tests/test_auth.py` | Two new tests: credentialed CORS only for listed origins, and the CLI byte-order-mark fix. |

Not touched: the YOLO model, detection settings, the inference pipeline, CCTV configuration, invoice extraction (backend and frontend), the local-only safeguard, `ADMIN_API_TOKEN` handling, `main` and the snapshot branch. No dependencies were added.

## Test results

- **Backend (full suite, isolated storage):** `166 passed, 38 skipped`, exit 0.
- **TypeScript:** `tsc --noEmit` exit 0.
- **Production build:** `next build` compiled successfully, 24/24 pages. The build used the gitignored distDir `.next/buildcheck`, which was then removed. `tsconfig.json` and `next-env.d.ts` were restored and are unchanged.
- **38 skipped tests — unresolved coverage limitation.** All are skipped because the trained model or test images are not present in the test environment: integration 19, scheduler 8, lab_api 5, capture_faults 4, roi_upload 2. These paths were **not** exercised by automated tests.
- There is no frontend unit-test framework in the project, so frontend coverage comes from tsc, the build and the browser checks below.

## Browser checks (isolated stack)

The isolated stack:
- backend on 127.0.0.1:8767 with temporary storage and `LM_AUTH_MODE=on`;
- frontend `next dev` on localhost:3020;
- one synthetic operator account.

All of it was removed afterwards.

1. **Wrong password:** shows "Invalid username or password." and clears the password field.
2. **Correct password:** opens the dashboard with the user chip. `/me` reports a session. `document.cookie` is empty (the cookie is HttpOnly). localStorage and sessionStorage are empty.
3. **Page refresh:** stays signed in.
4. **Navigation:** Image Detection, Alerts and Invoice pages load. A credentialed `POST /api/detect/image` returned 200. `POST /api/invoices/extract` returned 200.
5. **Sign out:** shows "You have been signed out." and `/me` reports unauthenticated.
6. **Session ended on the server** (account disabled, then re-enabled): shows "Your session has expired. Please sign in again."
7. **Lock-out after 5 failures:** shows "Too many failed sign-in attempts. Try again in about 15 minutes."
8. **`LM_AUTH_MODE=off`:** the app works as before, with no login screen and no sign-out chip, and the invoice page works. No `auth.db` was created.
9. **Leak check:** no password or session ID appeared in the browser console, the backend log or the frontend log.

## Integrity checks

- Live database fingerprint `f7c83c2dafd7ea24` is unchanged.
- No real `storage/auth.db` exists.
- Model sha256 `d1de9cb6…62bc6` matches the Step 0 baseline.
- No changed file touches settings, detection, CCTV or invoice code.
- Branch pointers: `main` 95c151b and the snapshot d0ea84d are unchanged. Nothing was pushed.
- Scan of the changed files found no real secrets and no real LAN address. The only matches are the synthetic placeholders `192.168.1.20` and `test-admin-token` in tests.

## Final verification (after the invoice removal, on top of 265e98a)

- **Backend (full suite, isolated storage):** `133 passed, 38 skipped`, exit 0. The 38 skipped tests are the same as above: 37 with "trained model/test images not present" and 1 with "no test images".
- **TypeScript:** `tsc --noEmit` exit 0.
- **Production build:** compiled successfully, 23/23 pages, with no invoice route.
- **No invoice code reintroduced:** no invoice route, import or link. The only matches are the `.gitignore` rules and the guard test `tests/test_app_routes.py`.
- **Browser checks (isolated stack, synthetic account, all removed afterwards):**
  - a wrong password shows "Invalid username or password." and clears the password field;
  - sign-in works, `document.cookie` is empty, and localStorage and sessionStorage are empty;
  - the session survives a page refresh and navigation (Alerts, Image Detection, Analytics);
  - a credentialed `POST /api/detect/image` returned 200 with model `v3_store_cctv`, and the live-event stream opened with credentials;
  - `/api/alerts`, `/api/cameras` and `/api/analytics` returned 200;
  - ending the session on the server shows "Your session has expired. Please sign in again.";
  - Sign out shows "You have been signed out.", and `/me` reports unauthenticated;
  - with `LM_AUTH_MODE` unset (the default): `/me` reports off, login returns 404, there is no login screen and no Sign out button, the dashboard and alerts work, and no `auth.db` is created.
- **Cookie:** `ldm_session` is `HttpOnly; SameSite=strict; Path=/api; Max-Age=43200`. `Secure` is absent over plain HTTP, as expected for `auto`. Only the listed origin receives `Access-Control-Allow-Origin`; another origin does not.
- **Leak check:** the test password, the wrong password and the session cookie appear in no backend or frontend log and nowhere in the browser console.
- **Integrity:**
  - model sha256 and live DB fingerprint `f7c83c2dafd7ea24` are unchanged, and the live DB still has 0 settings rows;
  - no real `storage/auth.db`;
  - no detection, CCTV, alerts, analytics, settings, local-only or auth-service file is changed;
  - `main` and the snapshot branch are unchanged, and nothing was pushed.

## Remaining security limitations

1. **No route-level protection.** Every API route except `/api/auth/*` is still open to local callers, whether or not anyone is signed in. The login gate is UI-only (Step 4).
2. The admin token, if used, is still kept in browser sessionStorage. This is pre-existing behaviour.
3. CSRF protection relies on SameSite=Strict only. There is no Origin/Referer check (planned for Step 4).
4. Rate limiting covers login only.
5. The UI must be opened on the same host name as the API (e.g. `localhost` on both). Otherwise SameSite blocks the cookie.
6. 38 skipped tests (see above). Frontend testing is manual and in the browser only.
7. Decisions D1 and D3–D7 (roles, export permission, session lengths, legacy token, rate limits, health detail) are still needed before Step 4. D2 (invoice protection) no longer applies, because invoice extraction was removed.
