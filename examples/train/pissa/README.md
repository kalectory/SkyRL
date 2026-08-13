# PiSSA with Megatron

PiSSA training uses two endpoints. The producer decomposes the original model once and writes a paired residual base and Megatron checkpoint. Training then uses the residual base with ordinary LoRA initialization and loads the checkpoint before its first update or sample.

## Produce the artifacts

Start the producer:

```bash
bash examples/train/pissa/run_qwen2_5_0.5b_megatron_pissa_producer.sh
```

Create a model and save both outputs with a Tinker client:

```python
import tinker

service_client = tinker.ServiceClient(base_url="http://localhost:8000", api_key="tml-dummy")
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    rank=32,
)
adapter_checkpoint = training_client.save_state("pissa_adapter").result().path
residual_base = training_client.save_weights_for_sampler("pissa_residual_base").result().path
print(adapter_checkpoint, residual_base)
```

Publish or mount the residual-base export where it can be used as a model path. Keep the adapter checkpoint in a checkpoint store visible to the training endpoint.

## Train from the artifacts

Start the training endpoint with the published residual model:

```bash
PISSA_RESIDUAL_MODEL=/path/to/pissa_residual_base \
  bash examples/train/pissa/run_qwen2_5_0.5b_megatron_pissa.sh
```

The training client uses the normal checkpoint-resume path:

```python
service_client = tinker.ServiceClient(base_url="http://localhost:8000", api_key="tml-dummy")
training_client = service_client.create_lora_training_client(
    base_model="/path/to/pissa_residual_base",
    rank=32,
)
training_client.load_state(adapter_checkpoint)
```

Do not set `init_method=pissa` on the training endpoint. Its default Kaiming adapter is only a correctly shaped container; `load_state` replaces it before training or sampling. The residual base and checkpoint must use the same source model, rank, alpha, and target modules.
