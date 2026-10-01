"""Optional native model runtime: CPU PEFT adapters under Identity authorization.

Install with `pip install .`, then run
`uvicorn ruvllm_microlora_runtime.app:create_app --factory --host 127.0.0.1`.
Required environment: IDENTITY_ISSUER, IDENTITY_JWKS_URL (HTTPS),
IDENTITY_EXPECTED_ACTOR, MICROLORA_ADAPTER_ROOT (private 0700 directory).
Optional MICROLORA_COMPUTE_USD_PER_TOKEN defaults to zero for local CPU;
this tariff accounts for runtime tokens, not any remote provider billing.
TLS termination and deployment authorization belong to the operator.
"""
