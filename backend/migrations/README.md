# SignalBridge database migrations

Versioned migrations in `versions/` expose `upgrade(connection)` and
`downgrade(connection)` functions using the project's existing SQLAlchemy
dependency. Production deployments should record applied versions in their
normal deployment system.

