class GPTNeoConfig(object):

    def __init__(
        self,
        block_size = 1024,
        window_size = 256,
        vocab_size = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer = 12,
        attention_types = [["global", "local"], 6],
        n_head = 12,
        n_embd = 768,
        dropout = 0.0,
        bias = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ):
        self.block_size = block_size
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.num_layers = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        self.attention_types = attention_types
        self.attention_layers = self.expand_attention_types_params(attention_types)

        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for _ in range(attention_types[1]):
            attentions.extend(attention_types[0])
        return attentions
    
    @property
    def _attn_implementation(self):
        # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value
