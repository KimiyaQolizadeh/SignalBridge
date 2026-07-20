import type { PropsWithChildren } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

export function AppLayout({ children }: PropsWithChildren) {
  const { pathname } = useLocation()
  const transcriptsActive = pathname === '/' || pathname.startsWith('/transcripts/')

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header className="app-header">
        <div className="header-content">
          <div className="brand-lockup">
            <div className="brand-mark" aria-hidden="true">
              SB
            </div>
            <div className="brand-copy">
              <h1>SignalBridge</h1>
              <span>Meeting Intelligence</span>
            </div>
            <span className="internal-badge">Internal</span>
          </div>
          <nav className="app-nav" aria-label="Primary navigation">
            <NavLink
              to="/"
              end
              aria-current={transcriptsActive ? 'page' : undefined}
              className={'app-nav__link' + (transcriptsActive ? ' active' : '')}
            >
              Transcript workspace
            </NavLink>
            <NavLink
              to="/upload"
              className={({ isActive }) =>
                'button button--primary app-nav__upload' +
                (isActive ? ' active' : '')
              }
            >
              Upload transcript
            </NavLink>
          </nav>
        </div>
      </header>
      <main id="main-content" className="main-content" tabIndex={-1}>
        {children}
      </main>
      <footer className="app-footer">
        <div>
          SignalBridge supports internal analysis and decision-making. Results
          should be reviewed before use.
        </div>
      </footer>
    </div>
  )
}
