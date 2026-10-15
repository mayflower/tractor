# API and normalization fixtures

These authored cases cover normalized quantities, record association, original
source offsets and HTTP validation errors. Run `just check` to validate their
structure and independent arithmetic gold. Fixture checks alone do not measure
model extraction accuracy; use `just check-model models/tractor cpu`.
