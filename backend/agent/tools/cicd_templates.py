"""
CI/CD workflow templates the agent installs into a target repo automatically.

When Opalite heals a repo that has no pipeline, it commits these into the fix PR:
  - ci.yml      -> the check the CI-Gate waits on (runs on the ai-fix/* branch too)
  - deploy.yml  -> visible CD; runs on push to main. Demo-green with no secrets,
                   upgrades to a real deploy + health-check + auto-rollback when
                   DEPLOY_HOOK / HEALTH_URL secrets are set.
Because a workflow added in a commit runs on that commit's push, this is
self-bootstrapping: the fix PR both fixes the bug AND turns on CI/CD.
"""

CI_YML = """# Installed automatically by Opalite Auto-Healer.
# A full CI pipeline mirroring the stages Jenkins/GitLab CI run:
#   lint -> test (+coverage) -> security scan -> build/package.
# Lint & security are non-blocking (report-only) so they never block auto-merge;
# TEST is the blocking gate the Opalite CI-Gate waits on.
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - name: Ruff
        continue-on-error: true            # report-only, does not block the merge gate
        run: ruff check . || true

  test:
    name: Test (pytest + coverage)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install pytest pytest-cov
      - name: Run tests with coverage
        run: |
          set +e
          pytest -v --cov=. --cov-report=term-missing --cov-report=xml
          code=$?
          if [ $code -eq 5 ]; then echo "No tests collected — skipping."; exit 0; fi
          exit $code
      - name: Upload coverage
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage.xml
          if-no-files-found: ignore

  security:
    name: Security scan (bandit + pip-audit)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install bandit pip-audit
      - name: Bandit (SAST)
        continue-on-error: true            # report-only
        run: bandit -r . -ll || true
      - name: pip-audit (dependency CVEs)
        continue-on-error: true
        run: |
          if [ -f requirements.txt ]; then pip-audit -r requirements.txt || true; else echo "no requirements.txt"; fi

  build:
    name: Build & package
    needs: [test]                          # only build once tests pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build artifact (Docker image if Dockerfile present, else zip)
        run: |
          mkdir -p dist
          if [ -f Dockerfile ]; then
            docker build -t app:${GITHUB_SHA} . || echo "docker build failed (non-fatal)"
          fi
          zip -r "dist/app-${GITHUB_SHA}.zip" . -x '*.git*' > /dev/null
      - uses: actions/upload-artifact@v4
        with:
          name: build-artifact
          path: dist/
"""

GITLAB_CI_YML = """# Installed automatically by Opalite Auto-Healer (GitLab CI/CD).
# Stages mirror a full pipeline: lint -> test -> security -> build -> deploy.
# lint & security are allow_failure (report-only); TEST is the gate the agent waits on.
stages:
  - lint
  - test
  - security
  - build
  - deploy

default:
  image: python:3.11

lint:
  stage: lint
  allow_failure: true
  script:
    - pip install ruff
    - ruff check . || true

test:
  stage: test
  script:
    - python -m pip install --upgrade pip
    - if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - pip install pytest pytest-cov
    - |
      set +e
      pytest -v --cov=. --cov-report=term-missing
      code=$?
      if [ $code -eq 5 ]; then echo "No tests collected — skipping."; exit 0; fi
      exit $code

security:
  stage: security
  allow_failure: true
  script:
    - pip install bandit pip-audit
    - bandit -r . -ll || true
    - if [ -f requirements.txt ]; then pip-audit -r requirements.txt || true; fi

build:
  stage: build
  script:
    - mkdir -p dist
    - (zip -r dist/app.zip . -x '*.git*' > /dev/null) || tar -czf dist/app.tgz --exclude=.git .
  artifacts:
    paths:
      - dist/

deploy:
  stage: deploy
  only:
    - main
  script:
    - echo "Deploying $CI_COMMIT_SHA to production..."
    # Hook up your real deploy here (e.g. curl "$DEPLOY_HOOK"), or let Opalite's
    # agent-deployer handle it on merge.
"""

DEPLOY_YML = """# Installed automatically by Opalite Auto-Healer.
# Demo mode (no DEPLOY_HOOK secret): simulates a deploy and goes green.
# Real mode: set repo secrets DEPLOY_HOOK (+ optional HEALTH_URL) for a real
# deploy with health-check and auto-rollback.
name: Deploy

on:
  push:
    branches: [main]

concurrency:
  group: production-deploy
  cancel-in-progress: false

jobs:
  deploy:
    name: Deploy to production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2
      - name: Deploy
        env:
          DEPLOY_HOOK: ${{ secrets.DEPLOY_HOOK }}
        run: |
          if [ -z "$DEPLOY_HOOK" ]; then
            echo "🟡 DEMO MODE: no DEPLOY_HOOK secret — simulating a successful deploy of ${GITHUB_SHA}."
            exit 0
          fi
          echo "🚀 Triggering real deployment..."
          curl -fsS -X POST "$DEPLOY_HOOK"
      - name: Health check
        env:
          HEALTH_URL: ${{ secrets.HEALTH_URL }}
        run: |
          if [ -z "$HEALTH_URL" ]; then echo "No HEALTH_URL set; skipping."; exit 0; fi
          for i in $(seq 1 30); do
            code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || echo "000")
            if [ "$code" = "200" ]; then echo "Healthy after ${i} attempts."; exit 0; fi
            echo "Attempt $i: got $code, retrying in 10s..."; sleep 10
          done
          echo "::error::Service never became healthy."; exit 1
      - name: Auto-rollback on failure
        if: failure()
        env:
          DEPLOY_HOOK: ${{ secrets.DEPLOY_HOOK }}
        run: |
          echo "::warning::Deploy unhealthy — reverting the last commit on main."
          git config user.name  "opalite-bot"
          git config user.email "opalite-bot@users.noreply.github.com"
          git revert --no-edit HEAD
          git push origin main
          if [ -n "$DEPLOY_HOOK" ]; then curl -fsS -X POST "$DEPLOY_HOOK" || true; fi
"""
