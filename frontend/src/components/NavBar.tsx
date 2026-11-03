import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Photo' },
  { to: '/video', label: 'Video' },
  { to: '/live-camera', label: 'Camera' },
]

export default function NavBar() {
  return (
    <header className="sticky top-0 z-10 border-b border-line bg-paper/90 backdrop-blur">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-3 px-4 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex min-w-0 items-center gap-2">
          <span className="text-lg" aria-hidden>
            👥
          </span>
          <span className="truncate text-[15px] font-bold tracking-tight text-ink">
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
                  'rounded-md px-3 py-1.5 text-sm transition-colors',
                  isActive
                    ? 'bg-ink font-semibold text-paper'
                    : 'text-muted hover:bg-soft hover:text-ink',
                ].join(' ')
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
