import torch
import torch.nn as nn
from network_utils import build_mlp


def test_basic_functionality():
    """Test basic MLP creation and forward pass"""
    print("=== Test 1: Basic Functionality ===")

    model = build_mlp(input_size=5, output_size=3, n_layers=2, size=10)

    # Test forward pass
    x = torch.randn(1, 5)  # batch_size=1, input_size=5
    output = model(x)

    assert output.shape == (1, 3), f"Expected output shape (1, 3), got {output.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    print(f"✓ Model has {len(list(model.parameters()))} parameter tensors")
    print()


def test_different_architectures():
    """Test various MLP architectures"""
    print("=== Test 2: Different Architectures ===")

    test_cases = [
        (4, 2, 1, 8),  # Single hidden layer
        (10, 5, 3, 20),  # Three hidden layers
        (100, 1, 4, 50),  # Deep network for binary classification
        (2, 10, 2, 32),  # Small input, large output
    ]

    for input_size, output_size, n_layers, size in test_cases:
        model = build_mlp(input_size, output_size, n_layers, size)

        # Test with batch of data
        x = torch.randn(5, input_size)  # batch_size=5
        output = model(x)

        expected_shape = (5, output_size)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

        print(
            f"✓ Architecture ({input_size}->{size}x{n_layers}->{output_size}): {x.shape} -> {output.shape}"
        )
    print()


def test_layer_structure():
    """Test that the model has the correct layer structure"""
    print("=== Test 3: Layer Structure ===")

    model = build_mlp(input_size=4, output_size=2, n_layers=2, size=6)

    # Expected structure: Linear -> ReLU -> Linear -> ReLU -> Linear
    expected_layers = [
        nn.Linear,  # First hidden layer
        nn.ReLU,  # First activation
        nn.Linear,  # Second hidden layer
        nn.ReLU,  # Second activation
        nn.Linear,  # Output layer (no activation)
    ]

    actual_layers = list(model.children())

    assert len(actual_layers) == len(expected_layers), (
        f"Expected {len(expected_layers)} layers, got {len(actual_layers)}"
    )

    for i, (actual, expected_type) in enumerate(zip(actual_layers, expected_layers)):
        assert isinstance(actual, expected_type), (
            f"Layer {i}: expected {expected_type.__name__}, got {type(actual).__name__}"
        )

    print(f"✓ Correct layer structure with {len(actual_layers)} layers")

    # Check layer dimensions
    linear_layers = [layer for layer in actual_layers if isinstance(layer, nn.Linear)]
    print(
        f"✓ Linear layer dimensions: {[f'{layer.in_features}->{layer.out_features}' for layer in linear_layers]}"
    )
    print()


def test_gradient_flow():
    """Test that gradients flow properly through the network"""
    print("=== Test 5: Gradient Flow ===")

    model = build_mlp(input_size=4, output_size=1, n_layers=2, size=8)

    # Create input and target
    x = torch.randn(3, 4, requires_grad=True)  # batch_size=3
    target = torch.randn(3, 1)

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"

    print("✓ Gradients computed successfully for all parameters")
    print(f"✓ Loss value: {loss.item():.4f}")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=== Test 6: Edge Cases ===")

    # Test with n_layers=1 (minimum)
    model1 = build_mlp(input_size=3, output_size=2, n_layers=1, size=5)
    x1 = torch.randn(2, 3)
    output1 = model1(x1)
    assert output1.shape == (2, 2), "Single layer test failed"
    print("✓ Single hidden layer (n_layers=1) works")

    # Test with large batch size
    model2 = build_mlp(input_size=5, output_size=3, n_layers=2, size=10)
    x2 = torch.randn(1000, 5)  # Large batch
    output2 = model2(x2)
    assert output2.shape == (1000, 3), "Large batch test failed"
    print("✓ Large batch size (1000) works")

    # Test with size=1 (minimum hidden layer size)
    model3 = build_mlp(input_size=2, output_size=1, n_layers=2, size=1)
    x3 = torch.randn(1, 2)
    output3 = model3(x3)
    assert output3.shape == (1, 1), "Minimum size test failed"
    print("✓ Minimum hidden layer size (size=1) works")
    print()


if __name__ == "__main__":
    print("Running tests for build_mlp function...\n")

    try:
        test_basic_functionality()
        test_different_architectures()
        test_layer_structure()
        test_gradient_flow()
        test_edge_cases()

        print("🎉 All tests passed! The build_mlp function is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise
