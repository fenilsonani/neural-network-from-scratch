#!/usr/bin/env python3
"""
CNN Layers Example: Comprehensive demonstration of convolutional layers.

This example showcases:
- Conv1D, Conv2D, Conv3D layers
- ConvTranspose layers for upsampling
- Advanced pooling operations
- Spatial dropout for regularization
- BatchNorm3D for normalization

Run with: python examples/cnn_layers_example.py
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_arch.core import Tensor
from neural_arch.nn import (
    Conv1d, Conv2d, Conv3d,
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    SpatialDropout1d, SpatialDropout2d, SpatialDropout3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d,
    AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    GlobalAvgPool1d, GlobalAvgPool2d,
    Sequential
)
from neural_arch.functional import relu


def demo_conv1d():
    """Demonstrate 1D convolution for time series or sequence data."""
    print("🔹 1D Convolution Demo (Time Series Processing)")
    print("-" * 50)
    
    # Create a simple 1D CNN for audio/time series processing
    batch_size, channels, length = 4, 1, 100
    
    # Input: (batch, channels, length) - e.g., audio waveform
    x = Tensor(np.random.randn(batch_size, channels, length), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Conv1D layers with different kernel sizes
    conv1 = Conv1d(1, 16, kernel_size=3, padding=1)  # Preserve length
    conv2 = Conv1d(16, 32, kernel_size=5, stride=2, padding=2)  # Downsample
    
    # Spatial dropout for regularization
    dropout = SpatialDropout1d(p=0.1)
    
    # Adaptive pooling
    adaptive_pool = AdaptiveAvgPool1d(25)  # Fixed output length
    global_pool = GlobalAvgPool1d()
    
    # Forward pass
    x = conv1(x)
    x = relu(x)
    print(f"After Conv1D(3): {x.shape}")
    
    x = conv2(x)
    x = relu(x)
    print(f"After Conv1D(5, stride=2): {x.shape}")
    
    x = dropout(x)
    print(f"After SpatialDropout1d: {x.shape}")
    
    x_adaptive = adaptive_pool(x)
    print(f"After AdaptiveAvgPool1d(25): {x_adaptive.shape}")
    
    x_global = global_pool(x)
    print(f"After GlobalAvgPool1d: {x_global.shape}")
    
    print("✅ 1D Convolution demo completed!\n")


def demo_conv2d():
    """Demonstrate 2D convolution for image processing."""
    print("🔹 2D Convolution Demo (Image Processing)")
    print("-" * 50)
    
    # Create a simple 2D CNN for image processing
    batch_size, channels, height, width = 2, 3, 32, 32
    
    # Input: (batch, channels, height, width) - e.g., RGB images
    x = Tensor(np.random.randn(batch_size, channels, height, width), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Conv2D layers
    conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
    conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    
    # Batch normalization
    bn1 = BatchNorm2d(64)
    bn2 = BatchNorm2d(128)
    
    # Spatial dropout
    dropout = SpatialDropout2d(p=0.1)
    
    # Pooling
    adaptive_pool = AdaptiveAvgPool2d((8, 8))
    global_pool = GlobalAvgPool2d()
    
    # Forward pass
    x = conv1(x)
    x = bn1(x)
    x = relu(x)
    print(f"After Conv2D(3x3) + BN + ReLU: {x.shape}")
    
    x = conv2(x)
    x = bn2(x)
    x = relu(x)
    print(f"After Conv2D(3x3, stride=2) + BN + ReLU: {x.shape}")
    
    x = dropout(x)
    print(f"After SpatialDropout2d: {x.shape}")
    
    x_adaptive = adaptive_pool(x)
    print(f"After AdaptiveAvgPool2d(8x8): {x_adaptive.shape}")
    
    x_global = global_pool(x)
    print(f"After GlobalAvgPool2d: {x_global.shape}")
    
    print("✅ 2D Convolution demo completed!\n")


def demo_conv3d():
    """Demonstrate 3D convolution for video processing."""
    print("🔹 3D Convolution Demo (Video Processing)")
    print("-" * 50)
    
    # Create a simple 3D CNN for video processing
    batch_size, channels, depth, height, width = 1, 3, 16, 32, 32
    
    # Input: (batch, channels, depth, height, width) - e.g., video frames
    x = Tensor(np.random.randn(batch_size, channels, depth, height, width), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Conv3D layers
    conv1 = Conv3d(3, 32, kernel_size=3, padding=1)
    conv2 = Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
    
    # Batch normalization
    bn1 = BatchNorm3d(32)
    bn2 = BatchNorm3d(64)
    
    # Spatial dropout
    dropout = SpatialDropout3d(p=0.1)
    
    # Forward pass
    x = conv1(x)
    x = bn1(x)
    x = relu(x)
    print(f"After Conv3D(3x3x3) + BN + ReLU: {x.shape}")
    
    x = conv2(x)
    x = bn2(x)
    x = relu(x)
    print(f"After Conv3D(stride=(1,2,2)) + BN + ReLU: {x.shape}")
    
    x = dropout(x)
    print(f"After SpatialDropout3d: {x.shape}")
    
    print("✅ 3D Convolution demo completed!\n")


def demo_transpose_convolutions():
    """Demonstrate transpose convolutions for upsampling."""
    print("🔹 Transpose Convolution Demo (Upsampling)")
    print("-" * 50)
    
    # 1D transpose convolution
    x1d = Tensor(np.random.randn(2, 32, 25), requires_grad=True)
    print(f"1D Input shape: {x1d.shape}")
    
    conv_transpose_1d = ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
    upsampled_1d = conv_transpose_1d(x1d)
    print(f"After ConvTranspose1d (2x upsample): {upsampled_1d.shape}")
    
    # 2D transpose convolution
    x2d = Tensor(np.random.randn(2, 64, 8, 8), requires_grad=True)
    print(f"2D Input shape: {x2d.shape}")
    
    conv_transpose_2d = ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
    upsampled_2d = conv_transpose_2d(x2d)
    print(f"After ConvTranspose2d (2x upsample): {upsampled_2d.shape}")
    
    # 3D transpose convolution
    x3d = Tensor(np.random.randn(1, 32, 8, 8, 8), requires_grad=True)
    print(f"3D Input shape: {x3d.shape}")
    
    conv_transpose_3d = ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
    upsampled_3d = conv_transpose_3d(x3d)
    print(f"After ConvTranspose3d (2x upsample): {upsampled_3d.shape}")
    
    print("✅ Transpose convolution demo completed!\n")


def demo_encoder_decoder_architecture():
    """Demonstrate a complete encoder-decoder architecture using our layers."""
    print("🔹 Encoder-Decoder Architecture Demo")
    print("-" * 50)
    
    # Simple U-Net style architecture for image segmentation
    batch_size, channels, height, width = 1, 3, 64, 64
    x = Tensor(np.random.randn(batch_size, channels, height, width), requires_grad=True)
    print(f"Input image: {x.shape}")
    
    # Encoder (downsampling path)
    encoder = Sequential(
        Conv2d(3, 32, kernel_size=3, padding=1),
        BatchNorm2d(32),
        Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 64->32
        BatchNorm2d(32),
        
        Conv2d(32, 64, kernel_size=3, padding=1),
        BatchNorm2d(64),
        Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 32->16
        BatchNorm2d(64),
        
        Conv2d(64, 128, kernel_size=3, padding=1),
        BatchNorm2d(128),
        Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 16->8
        BatchNorm2d(128),
    )
    
    # Process through encoder
    encoded = x
    print("Encoder forward pass:")
    for i, layer in enumerate(encoder.modules_list):
        if isinstance(layer, (Conv2d, ConvTranspose2d)):
            encoded = layer(encoded)
            print(f"  After layer {i} ({layer.__class__.__name__}): {encoded.shape}")
            encoded = relu(encoded)
        elif isinstance(layer, (BatchNorm2d, SpatialDropout2d)):
            encoded = layer(encoded)
    
    print(f"Encoded features: {encoded.shape}")
    
    # Decoder (upsampling path)
    decoder_conv1 = ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 8->16
    decoder_conv2 = ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 16->32
    decoder_conv3 = ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)   # 32->64
    final_conv = Conv2d(16, 1, kernel_size=1)  # Final segmentation map
    
    # Decoder forward pass
    print("Decoder forward pass:")
    decoded = decoder_conv1(encoded)
    print(f"  After decoder layer 1: {decoded.shape}")
    decoded = relu(decoded)
    
    decoded = decoder_conv2(decoded)
    print(f"  After decoder layer 2: {decoded.shape}")
    decoded = relu(decoded)
    
    decoded = decoder_conv3(decoded)
    print(f"  After decoder layer 3: {decoded.shape}")
    decoded = relu(decoded)
    
    output = final_conv(decoded)
    print(f"Final segmentation output: {output.shape}")
    
    print("✅ Encoder-decoder architecture demo completed!\n")


def demo_performance_tips():
    """Demonstrate performance optimization tips."""
    print("🔹 Performance Optimization Tips")
    print("-" * 50)
    
    # Tip 1: Use appropriate initialization
    conv = Conv2d(3, 64, kernel_size=3, weight_init="he_uniform")
    print("✓ Use He initialization for ReLU networks")
    
    # Tip 2: Use bias=False with batch normalization
    conv_bn = Conv2d(3, 64, kernel_size=3, bias=False)
    bn = BatchNorm2d(64)
    print("✓ Use bias=False when followed by BatchNorm")
    
    # Tip 3: Use spatial dropout instead of regular dropout for CNNs
    spatial_dropout = SpatialDropout2d(p=0.1)
    print("✓ Use SpatialDropout for convolutional layers")
    
    # Tip 4: Use adaptive pooling for flexible input sizes
    adaptive_pool = AdaptiveAvgPool2d((7, 7))
    print("✓ Use adaptive pooling for different input sizes")
    
    # Tip 5: Use groups for efficient convolutions
    depthwise_conv = Conv2d(64, 64, kernel_size=3, groups=64, padding=1)
    pointwise_conv = Conv2d(64, 128, kernel_size=1)
    print("✓ Use depthwise separable convolutions for efficiency")
    
    print("✅ Performance tips demo completed!\n")


def main():
    """Run all CNN layer demonstrations."""
    print("🚀 Neural Architecture Framework - CNN Layers Demo")
    print("=" * 60)
    print()
    
    try:
        demo_conv1d()
        demo_conv2d()
        demo_conv3d()
        demo_transpose_convolutions()
        demo_encoder_decoder_architecture()
        demo_performance_tips()
        
        print("🎉 All CNN layer demos completed successfully!")
        print("🔥 Your neural architecture framework is ready for computer vision!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()