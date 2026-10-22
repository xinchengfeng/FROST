import torch
from torch import nn

def fsm_stationary_basis(x: torch.Tensor, n_states: int) -> torch.Tensor:
    """Stationary state probabilities for the reflected N-state FSM.

    pi_i(x) is proportional to (x / (1-x))**i.  The log-domain form is
    numerically stable near the interval boundaries.
    """
    x = x.clamp(1e-5, 1.0 - 1e-5)
    log_ratio = torch.logit(x)
    indices = torch.arange(n_states, device=x.device, dtype=x.dtype)
    return torch.softmax(log_ratio.unsqueeze(-1) * indices, dim=-1)

class TTScalarOperator(nn.Module):
    """Analytical expected-output FROST operator for scalar mappings."""

    def __init__(self, input_dim: int, n_states: int = 4, rank: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.n_states = n_states
        self.rank = rank
        cores: list[nn.Parameter] = []
        cores.append(nn.Parameter(torch.randn(1, n_states, rank) * 0.15))
        for _ in range(1, input_dim - 1):
            core = torch.zeros(rank, n_states, rank)
            identity = torch.eye(rank)
            for state in range(n_states):
                core[:, state, :] = identity + 0.02 * torch.randn(rank, rank)
            cores.append(nn.Parameter(core))
        cores.append(nn.Parameter(torch.randn(rank, n_states, 1) * 0.15))
        self.cores = nn.ParameterList(cores)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = fsm_stationary_basis(x, self.n_states)
        vector = torch.einsum("bn,anr->br", probs[:, 0], self.cores[0])
        for mode in range(1, self.input_dim - 1):
            matrix = torch.einsum(
                "bn,anr->bar", probs[:, mode], self.cores[mode]
            )
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        final_matrix = torch.einsum(
            "bn,anr->bar", probs[:, -1], self.cores[-1]
        )
        raw = torch.bmm(vector.unsqueeze(1), final_matrix).squeeze(-1).squeeze(-1)
        return torch.sigmoid(self.output_scale * raw + self.output_bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

class TTAttentionOperator(nn.Module):
    """Multi-output FROST operator that maps a score row to an attention row."""

    def __init__(
        self,
        token_count: int,
        n_states: int = 4,
        rank: int = 12,
    ):
        super().__init__()
        self.token_count = token_count
        self.n_states = n_states
        self.rank = rank
        cores: list[nn.Parameter] = []
        cores.append(nn.Parameter(torch.randn(1, n_states, rank) * 0.15))
        for _ in range(1, token_count - 1):
            core = torch.zeros(rank, n_states, rank)
            identity = torch.eye(rank)
            for state in range(n_states):
                core[:, state, :] = identity + 0.02 * torch.randn(rank, rank)
            cores.append(nn.Parameter(core))
        cores.append(
            nn.Parameter(torch.randn(rank, n_states, token_count) * 0.15)
        )
        self.cores = nn.ParameterList(cores)
        self.output_bias = nn.Parameter(torch.zeros(token_count))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def expected_logits(self, x: torch.Tensor) -> torch.Tensor:
        probs = fsm_stationary_basis(x, self.n_states)
        vector = torch.einsum("bn,anr->br", probs[:, 0], self.cores[0])
        for mode in range(1, self.token_count - 1):
            matrix = torch.einsum(
                "bn,anr->bar", probs[:, mode], self.cores[mode]
            )
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        output_matrix = torch.einsum(
            "bn,ank->bak", probs[:, -1], self.cores[-1]
        )
        return torch.bmm(vector.unsqueeze(1), output_matrix).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.expected_logits(x) + self.output_bias
        temperature = self.temperature.clamp(0.2, 5.0)
        return torch.softmax(logits / temperature, dim=-1)

    def selected_logits(self, states: torch.Tensor) -> torch.Tensor:
        """TT lookup for a batch of explicit FSM state codewords."""
        batch_size = states.shape[0]
        first = self.cores[0][0, states[:, 0], :]
        vector = first
        for mode in range(1, self.token_count - 1):
            core = self.cores[mode]
            matrix = core[:, states[:, mode], :].permute(1, 0, 2)
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        last = self.cores[-1]
        output_matrix = last[:, states[:, -1], :].permute(1, 0, 2)
        return torch.bmm(vector.unsqueeze(1), output_matrix).squeeze(1)

    @torch.no_grad()
    def stochastic_forward(
        self,
        x: torch.Tensor,
        bitstream_length: int,
        burn_in: int = 32,
    ) -> torch.Tensor:
        """Finite-window FSM simulation followed by the learned output calibration."""
        batch_size, dimensions = x.shape
        states = torch.full(
            (batch_size, dimensions),
            fill_value=(self.n_states - 1) // 2,
            dtype=torch.long,
            device=x.device,
        )
        logits_sum = torch.zeros(
            batch_size, self.token_count, dtype=x.dtype, device=x.device
        )
        total_cycles = burn_in + bitstream_length
        for cycle in range(total_cycles):
            bits = torch.rand_like(x) < x
            states = torch.where(
                bits,
                torch.clamp(states + 1, max=self.n_states - 1),
                torch.clamp(states - 1, min=0),
            )
            if cycle >= burn_in:
                logits_sum += self.selected_logits(states)
        mean_logits = logits_sum / bitstream_length + self.output_bias
        temperature = self.temperature.clamp(0.2, 5.0)
        return torch.softmax(mean_logits / temperature, dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())