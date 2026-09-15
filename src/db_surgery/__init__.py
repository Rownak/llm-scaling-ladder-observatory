"""One-off maintenance scripts against `ladder.db` that don't belong in `ladder` itself.

Not part of the `ladder` package's stable interface and not wired into
`ladderctl` — these are rare, destructive, human-invoked operations (cache
invalidation after a scoring-logic change, etc.), each documented with what
it does and why in its own module docstring. Always support `--dry-run`.
"""
