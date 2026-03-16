# Repository Guidelines for Copilot

Setup environment:
```
scripts/dev_setup.sh
```

Run lint checks:
```
scripts/run_lint.sh
```

Run tests:
```
scripts/run_tests.sh
```

## Coding Standards

Data-oriented Design:
- Avoid Object Oriented Programming
- Separate data and logic. Use pure functions and dataclasses

When modifying code:
1. Run lint checks and fix errors and warnings.
2. Run tests and update tests if necessary
3. Update `README.md` if needed.
