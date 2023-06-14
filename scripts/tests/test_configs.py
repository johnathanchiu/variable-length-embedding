from vle.utils import instantiate_from_config

enc_config = {
    "target": "vle.modules.Encoder",
    "params": {
        "channels": 16,
        "in_layers": 64,
    },
}

enc = instantiate_from_config(enc_config)

print(enc)
