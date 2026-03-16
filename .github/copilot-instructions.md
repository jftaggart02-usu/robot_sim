# Repository Guidelines for Copilot

Setup environment:
```bash
bash ./scripts/dev_setup.sh
```

## Coding Standards

Data-oriented Design:
- Avoid Object Oriented Programming.
- Separate data and logic. Use pure functions and dataclasses.

When modifying code:
1. Run `bash ./scripts/run_lint.sh` and fix errors and warnings.
2. Run `bash ./scripts/run_tests.sh` and update tests if necessary.
3. Update documentation in `README.md` if necessary.
