# Best-practice advisories for rendered MedVision manifests (non-blocking).
#
# `warn` rules surface in conftest output but do not fail the build.
# Shared helpers (workload_kinds, main_containers) live in lib.rego.
package main

import rego.v1

# Encourage running as non-root (pod- or container-level).
warn contains msg if {
	workload_kinds[input.kind]
	not input.spec.template.spec.securityContext.runAsNonRoot
	some c in main_containers(input)
	not c.securityContext.runAsNonRoot
	msg := sprintf("%s/%s: container %q has no runAsNonRoot — consider a non-root securityContext", [input.kind, input.metadata.name, c.name])
}

# Liveness probe recommended for long-running containers.
warn contains msg if {
	workload_kinds[input.kind]
	some c in main_containers(input)
	not c.livenessProbe
	msg := sprintf("%s/%s: container %q has no livenessProbe", [input.kind, input.metadata.name, c.name])
}

# Readiness probe recommended so Services only route to ready pods.
warn contains msg if {
	workload_kinds[input.kind]
	some c in main_containers(input)
	not c.readinessProbe
	msg := sprintf("%s/%s: container %q has no readinessProbe", [input.kind, input.metadata.name, c.name])
}

# Workloads should carry standard recommended labels.
warn contains msg if {
	workload_kinds[input.kind]
	not input.metadata.labels["app.kubernetes.io/part-of"]
	msg := sprintf("%s/%s: missing recommended label app.kubernetes.io/part-of", [input.kind, input.metadata.name])
}
