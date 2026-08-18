# Trust model

Adapter modules are trusted local Python and execute with the invoking user's privileges. NeuralABI
does not sandbox them and never downloads or automatically trusts remote model code.

Plans, certificates, checkpoint metadata, shard indexes, tensor keys, and output paths are
untrusted. They are bounded, schema-checked data. Plans cannot contain Python expressions or
callables; checkpoint loading is SafeTensors-only; shard names are restricted to local filenames;
output writes use temporary paths and atomic replacement; and input checkpoints cannot be selected
as outputs.

Optimizer bundles sit on the same boundary: a strict bounded JSON manifest plus a SafeTensors tensor
store. NeuralABI never deserializes a PyTorch optimizer pickle. The trusted live-object exporter
runs only after application code has built or loaded the model and optimizer. Legacy
deserialization stays the caller's responsibility, outside NeuralABI.
