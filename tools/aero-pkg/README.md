# Aero Package Manager (`aero-pkg`)

The official build system and module orchestrator for the Aero language (v0.5.0+), authored entirely within pure Aero natively. It manages dependency constraints utilizing semantic version boundaries and structures nested monorepo `workspace` configurations securely leveraging `aero::collections::HashMap` and zero-cost abstraction wrappers.

## Core Commands
Run standard `aero` operations explicitly utilizing the build router:
```bash
# Create a new standalone Aero binary
aero new project_name

# Download dependencies mapped inside aero.toml and compile bounds
aero build

# Build target out natively and execute entry point
aero run

# Test all associated constraints
aero test

# Add semantic versioning module targets
aero add aeronum -v "0.1.0"

# Publish verified library bundles up to registry
aero publish
```

## Supported Manifest Model (`aero.toml`)
An updated universal blueprint configuration targeting the Aero modules:
```toml
[package]
name = "my_library"
version = "0.1.0"
authors = ["Engineer"]

[dependencies]
aeronum = "0.1.0"

[features]
gpu_target = ["cuda_backend"]

[workspace]
members = [
    "labs/gpu",
    "labs/ai"
]
```

## Architecture Integrations
Leveraging the `aero::io::File` capabilities to pull metadata and evaluating standard `aero::collections::HashMap` boundaries for zero-cost resolving logic explicitly without garbage collection pauses.

Bootstrap compiled from `tools/aero-pkg/src/main.aero`.
