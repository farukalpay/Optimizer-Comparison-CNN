import torch
from hypothesis import given, strategies as st

from main import SimpleCNN


@st.composite
def image_batches(draw):
    batch = draw(st.integers(min_value=1, max_value=4))
    channels = draw(st.sampled_from([1, 3]))
    if channels == 1:
        x = torch.randn(batch, 1, 28, 28)
    else:
        x = torch.randn(batch, 3, 32, 32)
    return x, channels


@given(image_batches())
def test_simplecnn_forward_output(image_data):
    x, channels = image_data
    model = SimpleCNN(num_classes=10, input_channels=channels)
    out = model(x)
    assert out.shape == (x.shape[0], 10)
    assert torch.isfinite(out).all()
