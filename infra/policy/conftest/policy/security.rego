# Security guardrails for rendered MedVision manifests (blocking).
#
# Evaluated per-document by conftest. `deny` rules fail the build.
# Shared helpers (workload_kinds, all_containers, main_containers) live in lib.rego.
package main

import rego.v1

# No mutable :latest image tags — pin everything.
deny contains msg if {
	workload_kinds[input.kind]
	some c in all_containers(input)
	endswith(c.image, ":latest")
	msg := sprintf("%s/%s: container %q must not use the ':latest' image tag", [input.kind, input.metadata.name, c.name])
}

# Every image must carry an explicit tag.
deny contains msg if {
	workload_kinds[input.kind]
	some c in all_containers(input)
	not contains(c.image, ":")
	msg := sprintf("%s/%s: container %q image %q has no explicit tag", [input.kind, input.metadata.name, c.name, c.image])
}

# No privileged containers.
deny contains msg if {
	workload_kinds[input.kind]
	some c in all_containers(input)
	c.securityContext.privileged == true
	msg := sprintf("%s/%s: container %q must not run privileged", [input.kind, input.metadata.name, c.name])
}

# No privilege escalation.
deny contains msg if {
	workload_kinds[input.kind]
	some c in all_containers(input)
	c.securityContext.allowPrivilegeEscalation == true
	msg := sprintf("%s/%s: container %q must set allowPrivilegeEscalation=false", [input.kind, input.metadata.name, c.name])
}

# No host namespace sharing.
deny contains msg if {
	workload_kinds[input.kind]
	some ns in ["hostNetwork", "hostPID", "hostIPC"]
	object.get(input.spec.template.spec, ns, false) == true
	msg := sprintf("%s/%s: %s must not be enabled", [input.kind, input.metadata.name, ns])
}

# Main containers must declare CPU + memory limits and requests.
deny contains msg if {
	workload_kinds[input.kind]
	some c in main_containers(input)
	not c.resources.limits
	msg := sprintf("%s/%s: container %q must declare resources.limits", [input.kind, input.metadata.name, c.name])
}

deny contains msg if {
	workload_kinds[input.kind]
	some c in main_containers(input)
	not c.resources.requests
	msg := sprintf("%s/%s: container %q must declare resources.requests", [input.kind, input.metadata.name, c.name])
}

# Rendered Secrets must not contain empty values.
deny contains msg if {
	input.kind == "Secret"
	some key, value in input.stringData
	value == ""
	msg := sprintf("Secret/%s: key %q is empty", [input.metadata.name, key])
}
