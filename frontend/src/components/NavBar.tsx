import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Photo', icon: '🖼️' },
  { to: '/video', label: 'Video', icon: '🎬' },
  { to: '/live-camera', label: 'Camera', icon: '📹' },
]

export default function NavBar() {
  return (
    <header className="border-b border-slate-200 bg-white shadow-sm">
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-3 px-4 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex min-w-0 items-center gap-2">
          <span className="text-xl" aria-hidden>
            👥
          </span>
          <span className="truncate text-lg font-bold tracking-tight text-slate-900">
            CrowdSense
          </span>
        </NavLink>
        <nav className="flex items-center gap-1" aria-label="Primary">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) =>
                [
                  'rounded-md px-2 py-2 text-xs font-medium transition-colors sm:px-3 sm:text-sm',
                  isActive
                    ? 'bg-brand-50 text-brand-700'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
                ].join(' ')
              }
            >
              <span className="mr-1" aria-hidden>
                {link.icon}
              </span>
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
