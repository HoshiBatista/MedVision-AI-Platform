{{/*
Common helpers for the MedVision chart.
*/}}

{{- define "medvision.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "medvision.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "medvision.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "medvision.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: medvision
{{- end -}}

{{/*
Per-component selector labels. Call with (dict "ctx" . "component" "auth-service").
*/}}
{{- define "medvision.selectorLabels" -}}
app.kubernetes.io/name: {{ include "medvision.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Full label set for pod templates — common labels + selector labels in one block
(so callers don't emit a duplicate app.kubernetes.io/instance key).
Call with (dict "ctx" . "component" "auth-service").
*/}}
{{- define "medvision.podLabels" -}}
helm.sh/chart: {{ printf "%s-%s" .ctx.Chart.Name .ctx.Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/managed-by: {{ .ctx.Release.Service }}
app.kubernetes.io/part-of: medvision
{{ include "medvision.selectorLabels" . }}
{{- end -}}

{{/*
Convert an internal service key (auth_service) to a DNS-safe k8s name
(auth-service). Used for Service names and in-cluster URLs.
*/}}
{{- define "medvision.svcName" -}}
{{- . | replace "_" "-" -}}
{{- end -}}

{{/*
Resolve a fully-qualified image ref for a component.
Call with (dict "ctx" . "image" "auth_service").
*/}}
{{- define "medvision.image" -}}
{{- $repo := printf "%s/%s" .ctx.Values.image.repository .image -}}
{{- printf "%s%s:%s" .ctx.Values.global.imageRegistry $repo .ctx.Values.image.tag -}}
{{- end -}}

{{- define "medvision.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "medvision.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "medvision.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "medvision.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Async (asyncpg) DATABASE_URL pointing at the in-cluster postgres service.
*/}}
{{- define "medvision.databaseUrl" -}}
{{- printf "postgresql+asyncpg://%s:$(POSTGRES_PASSWORD)@%s-postgres:%v/%s" .Values.postgres.user (include "medvision.fullname" .) .Values.postgres.port .Values.postgres.database -}}
{{- end -}}

{{- define "medvision.syncDatabaseUrl" -}}
{{- printf "postgresql+psycopg2://%s:$(POSTGRES_PASSWORD)@%s-postgres:%v/%s" .Values.postgres.user (include "medvision.fullname" .) .Values.postgres.port .Values.postgres.database -}}
{{- end -}}

{{- define "medvision.redisHost" -}}
{{- printf "%s-redis:%v" (include "medvision.fullname" .) .Values.redis.port -}}
{{- end -}}

{{/*
OpenTelemetry OTLP HTTP endpoint. Uses otel.exporterEndpoint when set; otherwise,
when Jaeger is enabled, points at the in-cluster Jaeger collector.
*/}}
{{- define "medvision.otelEndpoint" -}}
{{- if .Values.otel.exporterEndpoint -}}
{{- .Values.otel.exporterEndpoint -}}
{{- else if .Values.jaeger.enabled -}}
{{- printf "http://%s-jaeger:%v" (include "medvision.fullname" .) .Values.jaeger.otlpHttpPort -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}

{{/*
Shared (non-secret) application config — rendered into the ConfigMap and hashed
into a checksum annotation on the workloads. Defined as a named template (not a
file include) so it resolves when a single template is rendered in isolation
(e.g. under helm-unittest).
*/}}
{{- define "medvision.appConfig" -}}
ENVIRONMENT: {{ .Values.commonEnv.ENVIRONMENT | quote }}
LOG_LEVEL: {{ .Values.commonEnv.LOG_LEVEL | quote }}
AUTO_CREATE_TABLES: "false"
JWT_ALGORITHM: {{ .Values.auth.jwtAlgorithm | quote }}
ACCESS_TOKEN_EXPIRE_MINUTES: {{ .Values.auth.accessTokenExpireMinutes | quote }}
REFRESH_TOKEN_EXPIRE_DAYS: {{ .Values.auth.refreshTokenExpireDays | quote }}
ADMIN_USERNAME: {{ .Values.auth.adminUsername | quote }}
DOCS_ENABLED: {{ .Values.auth.docsEnabled | quote }}
STORAGE_ROOT: "/data/studies"
HEATMAP_OUTPUT_DIR: "/data/heatmaps"
MODEL_REPO_ROOT: "/models"
INFERENCE_BACKEND: {{ .Values.inference.backend | quote }}
TRITON_HTTP_URL: {{ printf "%s-triton:%v" (include "medvision.fullname" .) .Values.triton.httpPort | quote }}
TRITON_GRPC_URL: {{ printf "%s-triton:%v" (include "medvision.fullname" .) .Values.triton.grpcPort | quote }}
GRADCAM_SERVICE_URL: {{ printf "http://%s-gradcam-service:8004" (include "medvision.fullname" .) | quote }}
OLLAMA_URL: {{ printf "http://%s-ollama:%v" (include "medvision.fullname" .) .Values.ollama.port | quote }}
REDIS_URL: {{ printf "redis://%s/0" (include "medvision.redisHost" .) | quote }}
CELERY_BROKER_URL: {{ printf "redis://%s/1" (include "medvision.redisHost" .) | quote }}
CELERY_RESULT_BACKEND: {{ printf "redis://%s/2" (include "medvision.redisHost" .) | quote }}
LLM_MODEL_NAME: {{ .Values.llm.modelName | quote }}
LLM_MAX_NEW_TOKENS: {{ .Values.llm.maxNewTokens | quote }}
OTEL_TRACES_ENABLED: {{ .Values.otel.tracesEnabled | quote }}
OTEL_EXPORTER_OTLP_ENDPOINT: {{ include "medvision.otelEndpoint" . | quote }}
{{- end -}}
