{{/*
Expand the name of the chart.
*/}}
{{- define "contextmine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "contextmine.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "contextmine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "contextmine.labels" -}}
helm.sh/chart: {{ include "contextmine.chart" . }}
{{ include "contextmine.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "contextmine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "contextmine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
PostgreSQL host - returns internal service name or external host
*/}}
{{- define "contextmine.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgres" (include "contextmine.fullname" .) }}
{{- else }}
{{- .Values.postgresql.external.host }}
{{- end }}
{{- end }}

{{/*
PostgreSQL port
*/}}
{{- define "contextmine.postgresql.port" -}}
{{- if .Values.postgresql.enabled }}
{{- print "5432" }}
{{- else }}
{{- .Values.postgresql.external.port | default "5432" }}
{{- end }}
{{- end }}

{{/*
Database URL for the main contextmine database
*/}}
{{- define "contextmine.databaseUrl" -}}
{{- $host := include "contextmine.postgresql.host" . }}
{{- $port := include "contextmine.postgresql.port" . }}
{{- printf "postgresql+asyncpg://%s:%s@%s:%s/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password $host $port .Values.postgresql.auth.database }}
{{- end }}

{{/*
Database URL for Prefect
*/}}
{{- define "contextmine.prefectDatabaseUrl" -}}
{{- $host := include "contextmine.postgresql.host" . }}
{{- $port := include "contextmine.postgresql.port" . }}
{{- printf "postgresql+asyncpg://%s:%s@%s:%s/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password $host $port .Values.postgresql.auth.prefectDatabase }}
{{- end }}

{{/*
Prefect API URL for workers to connect to
*/}}
{{- define "contextmine.prefectApiUrl" -}}
{{- printf "http://%s-prefect:%d/api" (include "contextmine.fullname" .) (.Values.prefect.service.port | int) }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "contextmine.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "contextmine.api.labels" -}}
{{ include "contextmine.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "contextmine.api.selectorLabels" -}}
{{ include "contextmine.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
PostgreSQL component labels
*/}}
{{- define "contextmine.postgres.labels" -}}
{{ include "contextmine.labels" . }}
app.kubernetes.io/component: postgres
{{- end }}

{{/*
PostgreSQL selector labels
*/}}
{{- define "contextmine.postgres.selectorLabels" -}}
{{ include "contextmine.selectorLabels" . }}
app.kubernetes.io/component: postgres
{{- end }}

{{/*
Prefect component labels
*/}}
{{- define "contextmine.prefect.labels" -}}
{{ include "contextmine.labels" . }}
app.kubernetes.io/component: prefect
{{- end }}

{{/*
Prefect selector labels
*/}}
{{- define "contextmine.prefect.selectorLabels" -}}
{{ include "contextmine.selectorLabels" . }}
app.kubernetes.io/component: prefect
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "contextmine.worker.labels" -}}
{{ include "contextmine.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "contextmine.worker.selectorLabels" -}}
{{ include "contextmine.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Secret name - returns existing secret name or generated one
*/}}
{{- define "contextmine.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "contextmine.fullname" .) }}
{{- end }}
{{- end }}
