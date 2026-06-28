# Shared helpers for the MedVision conftest policies.
# All policy files share `package main`, so these definitions are visible to
# security.rego and best_practices.rego (define them here exactly once).
package main

import rego.v1

workload_kinds := {"Deployment", "StatefulSet", "DaemonSet", "Job", "ReplicaSet"}

# All containers (init + main) for a workload document.
all_containers(obj) := cs if {
	workload_kinds[obj.kind]
	cs := array.concat(
		object.get(obj.spec.template.spec, "containers", []),
		object.get(obj.spec.template.spec, "initContainers", []),
	)
}

# Main (long-running) containers only.
main_containers(obj) := cs if {
	workload_kinds[obj.kind]
	cs := object.get(obj.spec.template.spec, "containers", [])
}
