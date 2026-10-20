{{- define "tractor.labels" -}}
app.kubernetes.io/name: tractor
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "tractor.selectorLabels" -}}
app.kubernetes.io/name: tractor
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "tractor.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}
