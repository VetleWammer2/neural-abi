# Trust model

Adapter modules are trusted local Python and execute with the invoking user's privileges. NeuralABI
does not sandbox them and never downloads or automatically trusts remote model code.

Plans, certificates, checkpoint metadata, shard indexes, tensor keys, and output paths are
untrusted. They are bounded, schema-checked data. Plans cannot contain Python expressions or
callables; checkpoint loading is SafeTensors-only; shard names are restricted to local filenames;
output writes use temporary paths and atomic replacement; and input checkpoints cannot be selected
as outputs.
