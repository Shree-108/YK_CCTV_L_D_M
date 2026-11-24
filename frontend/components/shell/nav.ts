import {
  Activity, AlertTriangle, Cctv, BarChart3, Boxes, Camera, ClipboardCheck, Database, FileBarChart, FileVideo,
  FlaskConical, History, ImageIcon, LifeBuoy, LayoutGrid, ListChecks, Microscope, ScanSearch, Settings, Sparkles, Target, type LucideIcon,
} from 'lucide-react'

export type BadgeKey = 'online' | 'newAlerts' | 'cameras' | 'hardPending'

export interface NavItem { href: string; label: string; icon: LucideIcon; badge?: BadgeKey; description: string }
export interface NavGroup { label: string; items: NavItem[]; hidden?: boolean }

// The sidebar shows the core workflow (Workspace) and System pages. Advanced tools are hidden from the sidebar
// but remain reachable through the Ctrl+K command palette and in-page links.
export const NAV: NavGroup[] = [
  { label: 'Workspace', items: [
    { href: '/', label: 'Overview', icon: LayoutGrid, description: 'Litter events, trends, stores and recent analyses' },
    { href: '/detect/image', label: 'Image Analysis', icon: ImageIcon, description: 'Upload a CCTV image and detect litter' },
    { href: '/detect/video', label: 'Video Analysis', icon: FileVideo, description: 'Track litter through a whole CCTV video' },
    { href: '/alerts', label: 'Alerts', icon: AlertTriangle, badge: 'newAlerts', description: 'Detected litter incidents' },
    { href: '/checks', label: 'Results', icon: ListChecks, description: 'One row per camera check (clean, litter_detected, view_blocked) with the importer CSV fields' },
    { href: '/analytics', label: 'Analytics', icon: BarChart3, description: 'Detection trends and distributions' },
  ] },
  { label: 'System', items: [
    { href: '/settings', label: 'Settings', icon: Settings, description: 'Detection, alerts, appearance, security' },
  ] },
  // Not in the sidebar, but the pages, routes and features remain (Ctrl+K palette and in-page links).
  { label: 'Advanced', hidden: true, items: [
    { href: '/results', label: 'Results Dashboard', icon: FileBarChart, description: 'Four-screenshot video analyses: summary, CSV, monthly, history' },
    { href: '/live-cctv', label: 'Live CCTV', icon: Cctv, badge: 'online', description: 'Connect an authorized NVR/RTSP stream and detect litter live' },
    { href: '/cameras', label: 'Camera Integration', icon: Camera, badge: 'cameras', description: 'Stores, cameras and RTSP configuration' },
    { href: '/help', label: 'Help Center', icon: LifeBuoy, description: 'How to use the litter detection workspace' },
    { href: '/history', label: 'History', icon: History, description: 'Searchable detection log' },
    { href: '/test-lab', label: 'Test Lab', icon: Microscope, description: 'Compare models on the same input' },
    { href: '/model-lab', label: 'Model Lab', icon: Sparkles, description: 'Model cards, deployment and rollback' },
    { href: '/models', label: 'Models', icon: Boxes, description: 'Model comparison table' },
    { href: '/experiments', label: 'Experiments', icon: FlaskConical, description: 'Training runs and curves' },
    { href: '/evaluation', label: 'Evaluation', icon: Target, description: 'Metrics, PR curves and threshold lab' },
    { href: '/datasets', label: 'Dataset Lab', icon: Database, description: 'Dataset versions, stats and validation' },
    { href: '/ground-truth', label: 'Ground Truth', icon: ClipboardCheck, description: 'Review and edit annotations' },
    { href: '/hard-examples', label: 'Hard Examples', icon: ScanSearch, badge: 'hardPending', description: 'Mined failure cases for retraining' },
    { href: '/reports', label: 'Reports', icon: FileBarChart, description: 'Exports and evaluation reports' },
    { href: '/system', label: 'System Health', icon: Activity, description: 'Services and resources' },
    { href: '/live', label: 'Live cameras', icon: Camera, description: 'Live camera wall (requires real cameras)' },
  ] },
]

export const ALL_NAV = NAV.flatMap((g) => g.items.map((i) => ({ ...i, group: g.label })))

const ALIASES: Record<string, string> = { '/upload': '/detect/image' }

export function navFor(pathname: string) {
  const p = ALIASES[pathname] ?? pathname
  return ALL_NAV.filter((i) => i.href === '/' ? p === '/' : p === i.href || p.startsWith(`${i.href}/`))
    .sort((a, b) => b.href.length - a.href.length)[0]
}
