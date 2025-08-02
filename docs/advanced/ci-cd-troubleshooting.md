# CI/CD Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps developers and DevOps engineers diagnose and resolve common issues encountered in the Neural Architecture Framework CI/CD pipeline.

## Table of Contents

- [Quick Diagnosis Checklist](#quick-diagnosis-checklist)
- [Pipeline Failures](#pipeline-failures)
- [Test Issues](#test-issues)
- [Security Scan Problems](#security-scan-problems)
- [Performance Issues](#performance-issues)
- [Deployment Problems](#deployment-problems)
- [Environment Issues](#environment-issues)
- [Debugging Tools & Techniques](#debugging-tools--techniques)
- [Emergency Procedures](#emergency-procedures)

## Quick Diagnosis Checklist

When encountering CI/CD issues, start with this quick checklist:

### ⚡ Rapid Assessment (2-3 minutes)

1. **Check Recent Changes**
   - Review the last 2-3 commits for obvious issues
   - Check if similar issues occurred in recent PRs
   - Verify if the problem is environment-specific

2. **Review Workflow Status**
   - Check which specific jobs are failing
   - Look for patterns (all jobs vs. specific jobs)
   - Check if it's a timeout vs. actual failure

3. **Environment Quick Check**
   - Is this affecting all branches or just one?
   - Are similar repositories having issues?
   - Check GitHub Status page for platform issues

### 🔍 Deep Dive (5-10 minutes)

1. **Log Analysis**
   - Download and examine detailed logs
   - Look for error patterns and stack traces
   - Check for resource constraints or timeouts

2. **Configuration Verification**
   - Verify workflow YAML syntax
   - Check environment variables and secrets
   - Validate dependency specifications

## Pipeline Failures

### Job Failure Patterns

#### All Jobs Failing

**Symptoms**: All or most jobs in a workflow are failing

**Common Causes**:
- YAML syntax errors in workflow files
- Missing or incorrect secrets/environment variables
- GitHub Actions platform issues
- Repository permission problems

**Diagnosis Steps**:
```bash
# Check workflow syntax
yamllint .github/workflows/*.yml

# Validate specific workflow
act --dry-run -W .github/workflows/ci-main.yml

# Check for secret availability (if applicable)
gh secret list
```

**Solutions**:
1. **YAML Syntax Error**:
   ```yaml
   # ❌ Incorrect indentation
   jobs:
   test:
     runs-on: ubuntu-latest
   
   # ✅ Correct indentation
   jobs:
     test:
       runs-on: ubuntu-latest
   ```

2. **Missing Secrets**:
   ```bash
   # Add required secrets
   gh secret set PYPI_API_TOKEN --body "your-token"
   gh secret set TEST_PYPI_API_TOKEN --body "your-test-token"
   ```

3. **Permission Issues**:
   ```yaml
   # Add required permissions
   permissions:
     contents: read
     security-events: write
     actions: read
   ```

#### Specific Job Failures

**Symptoms**: Only certain jobs are failing consistently

**Common Causes**:
- Job-specific dependency issues
- Resource limitations (memory, disk space)
- Platform-specific problems
- Test instability

**Diagnosis Steps**:
```bash
# Check job-specific logs
gh run view <run-id> --log

# Check resource usage in logs
grep -i "memory\|disk\|space" job-logs.txt

# Check for platform-specific issues
grep -i "windows\|macos\|ubuntu" job-logs.txt
```

### Matrix Job Failures

**Symptoms**: Failures in specific matrix combinations

**Common Causes**:
- Python version compatibility issues
- OS-specific dependencies
- Platform-specific test failures

**Diagnosis Example**:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
  fail-fast: false  # Don't stop other matrix jobs
```

**Solutions**:
1. **Platform-Specific Exclusions**:
   ```yaml
   strategy:
     matrix:
       os: [ubuntu-latest, windows-latest, macos-latest]
       python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
       exclude:
         - os: windows-latest
           python-version: '3.8'  # Known compatibility issue
   ```

2. **Conditional Steps**:
   ```yaml
   - name: Windows-specific setup
     if: runner.os == 'Windows'
     run: |
       # Windows-specific commands
   ```

## Test Issues

### Test Failures

#### Flaky Tests

**Symptoms**: Tests that pass/fail inconsistently

**Common Causes**:
- Race conditions in concurrent code
- Time-dependent assertions
- External dependency issues
- Resource contention

**Diagnosis Steps**:
```bash
# Run tests multiple times locally
for i in {1..10}; do pytest tests/test_flaky_module.py; done

# Check for timing issues
grep -r "time.sleep\|asyncio.sleep" tests/

# Look for external dependencies
grep -r "requests\|urllib\|http" tests/
```

**Solutions**:
1. **Add Retry Logic**:
   ```python
   @pytest.mark.flaky(reruns=3, reruns_delay=2)
   def test_sometimes_fails():
       # Test implementation
   ```

2. **Mock External Dependencies**:
   ```python
   @mock.patch('requests.get')
   def test_with_mocked_request(mock_get):
       mock_get.return_value.status_code = 200
       # Test implementation
   ```

3. **Add Timeout Protection**:
   ```python
   @pytest.mark.timeout(30)
   def test_with_timeout():
       # Test implementation
   ```

#### Memory-Related Test Failures

**Symptoms**: Tests failing with memory errors or OOM conditions

**Common Causes**:
- Large tensor operations in tests
- Memory leaks in test setup/teardown
- Insufficient CI runner memory

**Diagnosis Steps**:
```bash
# Check memory usage in tests
pytest tests/ --memray

# Monitor memory during test execution
python -m memory_profiler -s "pytest tests/test_memory_heavy.py"
```

**Solutions**:
1. **Reduce Test Data Size**:
   ```python
   # ❌ Large test data
   def test_model():
       data = torch.randn(1000, 1000, 1000)  # Too large for CI
   
   # ✅ Smaller test data
   def test_model():
       data = torch.randn(10, 10, 10)  # Sufficient for testing
   ```

2. **Add Memory Cleanup**:
   ```python
   def test_with_cleanup():
       try:
           # Test implementation
           pass
       finally:
           gc.collect()  # Force garbage collection
   ```

### Coverage Issues

**Symptoms**: Coverage dropping below required thresholds

**Common Causes**:
- New code without corresponding tests
- Tests not covering edge cases
- Configuration issues

**Diagnosis Steps**:
```bash
# Generate detailed coverage report
pytest --cov=neural_arch --cov-report=html --cov-report=term-missing

# Check uncovered lines
coverage report --show-missing

# Identify files with low coverage
coverage report --skip-covered --show-missing
```

**Solutions**:
1. **Add Missing Tests**:
   ```python
   def test_edge_case():
       """Test edge case that was missing coverage."""
       # Implementation
   ```

2. **Update Coverage Configuration**:
   ```ini
   [tool.coverage.run]
   omit = [
       "*/tests/*",
       "*/venv/*",
       "*/.venv/*"
   ]
   ```

## Security Scan Problems

### False Positives

**Symptoms**: Security tools reporting issues that aren't actual vulnerabilities

**Common Causes**:
- Tool configuration issues
- Outdated security databases
- Legitimate code patterns flagged as suspicious

**Solutions**:
1. **Bandit False Positives**:
   ```python
   # Add bandit ignore comment
   password = get_password()  # nosec B105
   
   # Or update pyproject.toml
   [tool.bandit]
   skips = ["B101", "B601"]
   ```

2. **Update Baseline Files**:
   ```bash
   # Update detect-secrets baseline
   detect-secrets scan --update .secrets.baseline
   ```

### Dependency Vulnerabilities

**Symptoms**: Security scanners finding vulnerabilities in dependencies

**Common Causes**:
- Outdated dependencies with known vulnerabilities
- Transitive dependency issues
- New vulnerabilities discovered

**Diagnosis Steps**:
```bash
# Check specific vulnerabilities
safety check --json | jq '.vulnerabilities'

# Audit with multiple tools
pip-audit --desc
osv-scanner --format json .
```

**Solutions**:
1. **Update Dependencies**:
   ```bash
   # Update specific package
   pip install --upgrade package-name
   
   # Update all dependencies
   pip-review --auto
   ```

2. **Pin Secure Versions**:
   ```txt
   # requirements.txt
   requests>=2.31.0  # Fixed CVE-2023-32681
   urllib3>=2.0.4    # Fixed multiple CVEs
   ```

3. **Temporary Ignores** (use sparingly):
   ```bash
   # Safety ignore
   safety check --ignore 12345
   
   # pip-audit ignore
   pip-audit --ignore-vuln GHSA-xxxx-xxxx-xxxx
   ```

## Performance Issues

### Benchmark Failures

**Symptoms**: Performance benchmarks failing or showing regressions

**Common Causes**:
- Code changes introducing performance regressions
- CI runner performance variability
- Benchmark configuration issues

**Diagnosis Steps**:
```bash
# Run benchmarks locally
pytest tests/ -k benchmark --benchmark-json=results.json

# Compare with historical data
python scripts/compare_benchmarks.py results.json baseline.json

# Check system resources during benchmarks
top -p $(pgrep -f pytest)
```

**Solutions**:
1. **Adjust Benchmark Thresholds**:
   ```python
   @pytest.mark.benchmark(
       min_rounds=5,
       max_time=10.0,
       warmup=True
   )
   def test_performance():
       # Benchmark implementation
   ```

2. **Add Performance Monitoring**:
   ```yaml
   - name: Performance regression check
     run: |
       python scripts/check_regression.py \
         --threshold 10 \
         --baseline baseline.json \
         --current results.json
   ```

### Long-Running Pipelines

**Symptoms**: Pipelines taking excessively long to complete

**Common Causes**:
- Inefficient test execution
- Poor caching strategy
- Resource contention

**Solutions**:
1. **Optimize Test Execution**:
   ```yaml
   - name: Run tests in parallel
     run: |
       pytest tests/ -n auto --dist worksteal
   ```

2. **Improve Caching**:
   ```yaml
   - name: Cache dependencies
     uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
       restore-keys: |
         ${{ runner.os }}-pip-
   ```

## Deployment Problems

### Release Failures

**Symptoms**: Release workflows failing during deployment

**Common Causes**:
- Authentication issues with PyPI
- Version conflicts
- Missing release artifacts

**Diagnosis Steps**:
```bash
# Check PyPI credentials
twine check dist/*

# Verify package integrity
python -m build --check

# Test upload to Test PyPI first
twine upload --repository testpypi dist/*
```

**Solutions**:
1. **Fix Authentication**:
   ```bash
   # Update PyPI token
   gh secret set PYPI_API_TOKEN --body "pypi-xxx"
   ```

2. **Version Validation**:
   ```python
   # Ensure proper version bumping
   import setuptools_scm
   version = setuptools_scm.get_version()
   print(f"Current version: {version}")
   ```

### Documentation Deployment Issues

**Symptoms**: Documentation failing to build or deploy

**Common Causes**:
- Sphinx configuration errors
- Missing dependencies
- GitHub Pages configuration issues

**Solutions**:
1. **Fix Sphinx Issues**:
   ```bash
   # Test locally
   cd docs/sphinx
   make html SPHINXOPTS="-W"
   ```

2. **Check Dependencies**:
   ```bash
   pip install -e ".[docs]"
   ```

## Environment Issues

### Python Version Problems

**Symptoms**: Jobs failing on specific Python versions

**Common Causes**:
- Version-specific syntax or library issues
- Dependency compatibility problems

**Solutions**:
1. **Version-Specific Handling**:
   ```python
   import sys
   
   if sys.version_info >= (3.10):
       # Python 3.10+ specific code
       pass
   else:
       # Fallback for older versions
       pass
   ```

2. **Conditional Dependencies**:
   ```txt
   # requirements.txt
   typing-extensions; python_version<"3.8"
   importlib-metadata; python_version<"3.8"
   ```

### Operating System Issues

**Symptoms**: Platform-specific failures

**Common Causes**:
- Path separator differences
- Package availability differences
- Permission issues

**Solutions**:
1. **Cross-Platform Paths**:
   ```python
   import os
   from pathlib import Path
   
   # ❌ Platform-specific
   path = "src/neural_arch/core.py"
   
   # ✅ Cross-platform
   path = Path("src") / "neural_arch" / "core.py"
   ```

2. **Platform-Specific Steps**:
   ```yaml
   - name: Install Windows dependencies
     if: runner.os == 'Windows'
     run: |
       choco install something
   
   - name: Install macOS dependencies
     if: runner.os == 'macOS'
     run: |
       brew install something
   ```

## Debugging Tools & Techniques

### Local Debugging

#### Using Act for Local Testing

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflow locally
act -j test

# Run specific job
act -j unit-tests

# Debug with shell access
act -j test --shell
```

#### Local Environment Replication

```bash
# Create exact CI environment
python -m venv ci-env
source ci-env/bin/activate  # or ci-env\Scripts\activate on Windows

# Install exact dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run exact test commands
pytest tests/ -v --tb=short --strict-markers
```

### Remote Debugging

#### SSH Access to Runners

```yaml
# Add debugging step
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 30
```

#### Artifact Analysis

```bash
# Download artifacts
gh run download <run-id>

# Analyze logs
grep -r "ERROR\|FAILED" downloaded-artifacts/

# Check coverage reports
open downloaded-artifacts/coverage-report/index.html
```

### Advanced Debugging

#### Performance Profiling

```python
# Add profiling to tests
import cProfile
import pstats

def profile_test():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Test code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### Memory Debugging

```python
# Memory tracking
import tracemalloc

tracemalloc.start()

# Code to debug

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## Emergency Procedures

### Pipeline Outage Response

1. **Immediate Assessment** (< 5 minutes):
   - Check GitHub Status page
   - Verify if issue is repository-specific
   - Check for recent configuration changes

2. **Communication** (< 10 minutes):
   - Notify development team
   - Create incident tracking issue
   - Update status communications

3. **Mitigation** (< 30 minutes):
   - Implement temporary workarounds
   - Disable problematic workflows if necessary
   - Switch to manual processes if critical

4. **Resolution**:
   - Identify root cause
   - Implement permanent fix
   - Test thoroughly before re-enabling

### Critical Security Issue Response

1. **Immediate Actions**:
   - Stop ongoing deployments
   - Assess impact and scope
   - Implement temporary mitigations

2. **Investigation**:
   - Analyze security scan results
   - Verify vulnerability details
   - Assess potential data exposure

3. **Remediation**:
   - Apply security patches
   - Update dependencies
   - Re-run security scans
   - Document changes

### Rollback Procedures

#### Failed Release Rollback

```bash
# Revert PyPI release (if possible)
# Note: PyPI doesn't allow true rollbacks, only yanking

# Revert GitHub release
gh release delete v1.2.3
git push --delete origin v1.2.3

# Create hotfix release
git checkout v1.2.2
git checkout -b hotfix/v1.2.4
# Apply critical fixes
git tag v1.2.4
git push origin v1.2.4
```

#### Configuration Rollback

```bash
# Revert workflow changes
git revert <commit-hash>
git push origin main

# Or restore from backup
git checkout HEAD~1 -- .github/workflows/
git commit -m "Restore previous workflow configuration"
git push origin main
```

---

## Getting Help

### Internal Resources

- **DevOps Team**: devops@neural-arch.ai
- **Security Team**: security@neural-arch.ai
- **Platform Team**: platform@neural-arch.ai

### External Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Community Forum](https://github.community)
- [GitHub Status Page](https://www.githubstatus.com)

### Creating Support Tickets

When creating support tickets, include:

1. **Problem Description**: Clear description of the issue
2. **Reproduction Steps**: How to reproduce the problem
3. **Environment Details**: OS, Python version, workflow name
4. **Error Messages**: Complete error messages and stack traces
5. **Recent Changes**: Any recent changes that might be related
6. **Logs**: Relevant log files or workflow run URLs

---

*This troubleshooting guide is continuously updated based on common issues encountered in the Neural Architecture Framework CI/CD pipeline. If you encounter an issue not covered here, please contribute by documenting the solution.*