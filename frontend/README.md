# CrowdSense AI — Frontend

React + TypeScript (Vite) single-page app for the three supported tools:

1. **Photo Analysis** — person detection, annotated frame, density grid, static features
2. **Video Analysis** — window features and per-window anomaly scores (when a model exists)
3. **Live Camera** — browser camera frames analyzed one at a time

The UI never invents numbers: missing metrics render as `—`, counts are labelled
detector estimates, and speeds are px/frame (never m/s).

## Prerequisites

- Node.js 18+ and npm
- Python API running (see below)
- `yolov8n.pt` in the repository root (already used by the pipeline)

## Development

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to 127.0.0.1:8000
```

Start the API in another terminal from the repository root:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Production build

```powershell
cd frontend
npm run build      # type-checks, then emits ../backend/static
cd ..
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

`backend/static/` is gitignored build output. The API serves `/`, `/video`, and
`/live-camera` from `index.html`, so browser refreshes work on every route.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server with hot reload and `/api` proxy |
| `npm run build` | `tsc --noEmit` + production bundle to `backend/static` |
| `npm run typecheck` | TypeScript only |
| `npm run preview` | Preview the production bundle locally |

## API contract

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness |
| GET | `/api/model` | Bundle availability, algorithm, threshold, features, caveat |
| POST | `/api/detect-image` | Multipart `file`, `conf`, `imgsz` → boxes, grid, static features, annotated JPEG |
| POST | `/api/detect-video` | Multipart `file`, `max_frames`, `stride`, `conf`, `imgsz` → windows + optional predictions |
| POST | `/api/detect-frame` | Multipart `frame`, `conf`, `imgsz` → single-frame boxes |

Set `VITE_API_URL` only when the API is on a different origin; the default
empty value uses same-origin `/api` (or the dev proxy).

## Structure

```
src/
  App.tsx                 # routes: /, /video, /live-camera
  index.tsx               # React entry point
  types.ts                # API response types
  lib/api.ts              # typed fetch client
  components/
    NavBar.tsx
    UploadDropzone.tsx
    ModelBadge.tsx
    ResultsPanel.tsx      # image + video result views
  pages/
    PhotoAnalysis.tsx
    VideoAnalysis.tsx
    LiveCamera.tsx
```
