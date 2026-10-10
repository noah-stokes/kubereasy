# Kubereasy

Kubereasy separates model research and training from the production autoscaling runtime.

```text
.
├── model/                  # Existing simulator, policies, and training code
├── runtime/
│   ├── src/kubereasy_autoscaler/
│   │   ├── controller/     # Reconciliation and decision loop
│   │   ├── telemetry/      # Prometheus and Kubernetes metric collection
│   │   ├── features/       # History windows and feature preparation
│   │   ├── inference/      # Model loading and prediction
│   │   ├── safety/         # Bounds, cooldowns, fallbacks, and kill switch
│   │   └── scaling/        # Kubernetes /scale integration
│   ├── config/             # Runtime configuration
│   └── tests/              # Runtime unit and integration tests
├── contracts/              # Shared feature, decision, and model schemas
├── deploy/                 # Helm chart and raw Kubernetes manifests
├── observability/          # Dashboards and alert rules
├── examples/               # Example workloads and policies
├── tests/e2e/              # End-to-end cluster tests
├── docs/                   # Architecture notes and runbooks
└── scripts/                # Local development and release automation
```

