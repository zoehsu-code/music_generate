Vendored inference dependencies from https://github.com/minzwon/musicfm at
`b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`, under the included MIT LICENSE.

Local change: `MusicFM25Hz.__init__` accepts `config_path` so the caller can
provide a pinned, cached Conformer configuration. No architecture, feature
normalization, or pretrained parameters are changed. The wrapper loads the
checkpoint with `map_location="cpu", weights_only=True` before device transfer.
Flash attention is disabled; the unvendored training/Flash modules are unused.
