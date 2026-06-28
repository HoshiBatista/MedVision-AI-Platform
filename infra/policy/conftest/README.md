# Conftest / OPA policies

Rego policies enforced against the **rendered** Helm manifests by the
`helm-k8s` GitHub Actions pipeline (job: `policy`).

- `policy/security.rego` — **blocking** (`deny`): no `:latest`/untagged images,
  no privileged containers, no privilege escalation, no host namespaces,
  required resource requests/limits on main containers, no empty Secret values.
- `policy/best_practices.rego` — **advisory** (`warn`): non-root, liveness/
  readiness probes, recommended labels.

## Run locally

```bash
helm template medv infra/helm/medvision \
  -f infra/helm/medvision/ci/cpu-values.yaml > /tmp/render.yaml

conftest test --policy infra/policy/conftest/policy /tmp/render.yaml
```

`deny` failures exit non-zero; `warn` findings are printed but do not fail.
